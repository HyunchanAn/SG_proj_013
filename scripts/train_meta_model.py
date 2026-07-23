import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
import httpx
import re
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

PROJ_001_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "SG_proj_001"))
PROJ_001_MODELS = os.path.join(PROJ_001_DIR, "models")

def sanitize(name):
    return re.sub(r"[^a-zA-Z0-9\-]", "_", str(name))

def extract_substrate(val):
    if not isinstance(val, str): return None
    m = re.search(r'\*\(([^)]+)\)', val)
    if m:
        conds = m.group(1).split('/')
        if len(conds) > 0:
            return conds[-1].strip()
    return None

def extract_target_val(s):
    if pd.isna(s): return np.nan
    s = str(s)
    m = re.search(r'\(([\d.,~\s]+)\)', s)
    if m:
        nums_str = m.group(1).replace('~', ',')
        nums = []
        for part in nums_str.split(','):
            try: nums.append(float(part.strip()))
            except: pass
        if len(nums) > 0: return np.mean(nums)
    match = re.search(r'(-?\d+(?:\.\d+)?)', s)
    if match: return float(match.group(1))
    return np.nan

def run_meta_training():
    print("--- Cascade Meta-Model Retraining (Step 7) ---")
    
    tg_path = os.path.join(PROJ_001_MODELS, "model_tg.pkl")
    visc_path = os.path.join(PROJ_001_MODELS, "model_viscosity.pkl")
    adh_path = os.path.join(PROJ_001_MODELS, "model_adhesion.pkl")
    
    print("Loading base models from SG_proj_001...")
    with open(tg_path, "rb") as f: model_tg = pickle.load(f)
    with open(visc_path, "rb") as f: model_visc = pickle.load(f)
    with open(adh_path, "rb") as f: model_adh = pickle.load(f)
    with open(os.path.join(PROJ_001_MODELS, "feature_names.pkl"), "rb") as f:
        unified_feature_names = pickle.load(f)
    with open(os.path.join(PROJ_001_MODELS, "kmeans_model.pkl"), "rb") as f:
        kmeans = pickle.load(f)

    api_url = os.getenv("MODULE_004_API_URL", "http://MacBookPro-HC:8000")
    print(f"Fetching data from {api_url} ...")
    
    res_syn = httpx.get(f"{api_url}/experiments/synthesis", timeout=15.0)
    syn_data = res_syn.json()
    res_coat = httpx.get(f"{api_url}/experiments/coating?limit=10000", timeout=15.0)
    coat_data = res_coat.json()

    syn_rows = []
    for item in syn_data:
        row = {}
        for k, v in item.items():
            if k in ["monomers", "solvents", "initiators", "emulsifiers_additives"]:
                if v:
                    if isinstance(v, str):
                        try: v = json.loads(v)
                        except: v = []
                    for formula in v:
                        if isinstance(formula, dict):
                            name = formula.get("name")
                            wt = formula.get("wt_percent")
                            if name and wt is not None:
                                row[f"rec_{name}(wt%)"] = wt
            else:
                row[f"syn_{k}"] = v
        row["syn_점착제"] = item.get("adhesive_id")
        row["syn_Tg"] = item.get("tg")
        row["syn_점도(cP)"] = item.get("viscosity_cp")
        syn_rows.append(row)
    df_syn = pd.DataFrame(syn_rows)
    
    coat_rows = []
    for item in coat_data:
        row = {}
        row["syn_점착제"] = item.get("adhesive_id")
        row["test_점착력"] = item.get("adhesion")
        coat_rows.append(row)
    df_coat = pd.DataFrame(coat_rows)
    
    df = pd.merge(df_coat, df_syn, on="syn_점착제", how="inner")
    
    sub_map = {}
    response = httpx.get(f"{api_url}/adherends", timeout=10.0)
    for item in response.json():
        name = item.get("product_name")
        energy = item.get("surface_energy_md")
        roughness = item.get("roughness_md")
        if name and name != "nan":
            sub_map[name] = {}
            if energy is not None: sub_map[name]['energy'] = float(energy)
            if roughness is not None: sub_map[name]['roughness'] = float(roughness)

    df['parsed_substrate'] = df['test_점착력'].apply(extract_substrate)
    df['feat_sub_energy'] = df['parsed_substrate'].apply(lambda x: sub_map.get(x, {}).get('energy', float('nan')))
    df['feat_sub_roughness'] = df['parsed_substrate'].apply(lambda x: sub_map.get(x, {}).get('roughness', float('nan')))
    df['feat_sub_energy'] = df['feat_sub_energy'].fillna(df['feat_sub_energy'].median())
    df['feat_sub_roughness'] = df['feat_sub_roughness'].fillna(df['feat_sub_roughness'].median())

    recipe_features = [c for c in df.columns if c.startswith("rec_")]
    syn_exclude = ["syn_Tg", "syn_점도(cP)", "syn_측정_값", "syn_점착제", "syn_첨부파일", "syn_작업자", "syn_분류", "syn_설명", "syn_pH", "syn_초기 유화제 농도", "syn_반응시간", "syn_모노머(wt%)", "syn_용제(wt%)", "syn_개시제(wt%)", "syn_유화제 및 첨가제(wt%)"]
    syn_features = [c for c in df.columns if c.startswith("syn_") and c not in syn_exclude]
    domain_features = [c for c in df.columns if c.startswith("feat_")]
    
    feature_candidates = list(set(recipe_features + syn_features + domain_features))
    X_dict = {}
    for col in feature_candidates:
        series_data = df[col]
        clean_name = sanitize(col)
        if series_data.dtype == "object":
            X_dict[clean_name] = series_data.astype("category").cat.codes
        else:
            X_dict[clean_name] = pd.to_numeric(series_data, errors="coerce").fillna(0)
    
    X_clean = pd.DataFrame(X_dict)
    
    rec_cols_clean = [sanitize(c) for c in recipe_features if sanitize(c) in X_clean.columns]
    clusters = kmeans.predict(X_clean[rec_cols_clean].fillna(0))
    for i in range(5):
        X_clean[f"cluster_{i}"] = (clusters == i).astype(int)

    # Reorder columns to match original unified_feature_names exactly
    missing_cols = set(unified_feature_names) - set(X_clean.columns)
    for c in missing_cols:
        X_clean[c] = 0
    X_aligned = X_clean[unified_feature_names]

    y_adh = df["test_점착력"].apply(extract_target_val)
    valid_mask = y_adh.notna() & (y_adh > 0)
    
    X_train = X_aligned[valid_mask].copy()
    y_true = y_adh[valid_mask]
    
    print("Generating base predictions (OOF equivalent)...")
    raw_tg = df.loc[valid_mask, "syn_Tg"].apply(extract_target_val)
    raw_visc = df.loc[valid_mask, "syn_점도(cP)"].apply(extract_target_val)
    
    pred_tg_val = model_tg.predict(X_train)
    pred_visc_val = model_visc.predict(X_train)
    
    X_train["syn_Tg"] = raw_tg.fillna(pd.Series(pred_tg_val, index=X_train.index))
    X_train["syn_점도(cP)"] = raw_visc.fillna(pd.Series(pred_visc_val, index=X_train.index))
    
    pred_adh_val = model_adh.predict(X_train)
    
    Meta_X = pd.DataFrame({
        "pred_tg": pred_tg_val,
        "pred_visc": pred_visc_val,
        "pred_adh": pred_adh_val
    }, index=y_true.index)
    
    print("Training XGBoost Meta-Learner...")
    meta_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, random_state=42)
    meta_model.fit(Meta_X, y_true)
    
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, meta_model.predict(Meta_X))
    print(f"Meta Model Training Complete! R2 Score: {r2:.4f}")
    
    with open(os.path.join(MODEL_DIR, "meta_xgboost.pkl"), "wb") as f:
        pickle.dump(meta_model, f)
        
    # Copy base models for predictor serving
    import shutil
    shutil.copy(tg_path, MODEL_DIR)
    shutil.copy(visc_path, MODEL_DIR)
    shutil.copy(adh_path, MODEL_DIR)
    shutil.copy(os.path.join(PROJ_001_MODELS, "feature_names.pkl"), MODEL_DIR)
    shutil.copy(os.path.join(PROJ_001_MODELS, "kmeans_model.pkl"), MODEL_DIR)

    print("Cascade Trigger Complete. All models synchronized in SG_proj_013/models/")

if __name__ == "__main__":
    run_meta_training()
