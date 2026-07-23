import os
import pickle
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from huggingface_hub import HfApi

def convert_and_push():
    models_dir = "/Users/hyunchanan/Documents/GitHub/SG_proj_013/models"
    hf_repo_id = "chemahc94/sg-adhesion-models" 
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set. Please log in or set HF_TOKEN.")
    
    api = HfApi(token=token)
    
    # Ensure the repo exists (create it if not, private by default)
    try:
        api.create_repo(repo_id=hf_repo_id, private=True, exist_ok=True)
        print(f"Repository {hf_repo_id} ready.")
    except Exception as e:
        print(f"Warning: Could not create repo or verify. Proceeding to upload. {e}")
    
    base_features = 76
    meta_features = 3 # pred_tg, pred_visc, pred_adh
    
    # 1. Convert models
    model_paths = {
        "model_tg.onnx": ("model_tg.pkl", base_features),
        "model_viscosity.onnx": ("model_viscosity.pkl", base_features),
        "model_adhesion.onnx": ("model_adhesion.pkl", 78),
        "meta_xgboost.onnx": ("meta_xgboost.pkl", meta_features),
    }
    
    for onnx_name, (pkl_name, dim) in model_paths.items():
        pkl_path = os.path.join(models_dir, pkl_name)
        onnx_path = os.path.join(models_dir, onnx_name)
        
        print(f"Converting {pkl_name} to {onnx_name} (dim: {dim})...")
        with open(pkl_path, "rb") as f:
            xgb_model = pickle.load(f)
            
        # Fix for ONNX converting XGBoost models with string feature names
        if hasattr(xgb_model, 'get_booster'):
            booster = xgb_model.get_booster()
            if booster.feature_names is not None:
                booster.feature_names = [f"f{i}" for i in range(len(booster.feature_names))]
        
        initial_types = [('float_input', FloatTensorType([None, dim]))]
        onnx_model = onnxmltools.convert_xgboost(xgb_model, initial_types=initial_types)
        
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        print(f"Pushing {onnx_name} to HF Hub...")
        api.upload_file(
            path_or_fileobj=onnx_path,
            path_in_repo=onnx_name,
            repo_id=hf_repo_id
        )
    
    # 2. Push metadata models (kmeans, features)
    metadata_files = ["feature_names.pkl", "kmeans_model.pkl"]
    for m in metadata_files:
        m_path = os.path.join(models_dir, m)
        print(f"Pushing metadata {m} to HF Hub...")
        api.upload_file(
            path_or_fileobj=m_path,
            path_in_repo=m,
            repo_id=hf_repo_id
        )
        
    print("All ONNX models and metadata successfully pushed to HuggingFace!")

if __name__ == "__main__":
    convert_and_push()
