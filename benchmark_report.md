# Benchmark & E2E Test Report

- **Repository**: SG_proj_013
- **Date**: 2026-07-14 22:44:58

## 1. E2E Testing Summary
❌ **Status**: FAILED

### Test Logs (Snippet)
```text
            "xgboost_prediction": {"adhesion": 5.0, "viscosity": 98.0},
            "ir_gnn_features": [0.1, 0.2, 0.3],
            "current_iteration": 5
        }
        # Send current_iteration = 5, wait, the validator schema has: ge=1, le=5.
        # If current_iteration is 5, it's allowed by schema. If we increment it, the predictor handles > 5 inside verify_and_predict or the schema might fail if we set > 5 in request.
        # Let's test with current_iteration=5. But wait, in schemas.py:
        # current_iteration: int = Field(default=1, ge=1, le=5)
        # If the input has current_iteration: 6, FastAPI validation will raise 422 Unprocessable Entity.
        # If the input has current_iteration: 5, it passes schema validation. Then in predictor.py:
        # if data.current_iteration > 5:
        #     return VerificationResult(is_passed=False, ...)
        # But wait! If we check `predictor.py` line 8:
        # if data.current_iteration > 5:
        # Since current_iteration can only be up to 5 in schemas.py, data.current_iteration > 5 is actually impossible to reach via API if schema validation blocks it! Let's check schemas.py again.
        # Ah, schemas.py has: current_iteration: int = Field(default=1, ge=1, le=5, description="무한 루프 방지")
        # Yes, so current_iteration must be <= 5.
        # Let's test a valid request with current_iteration=5.
        res = client.post("/verify", json=payload)
>       assert res.status_code == 200
E       assert 404 == 200
E        +  where 404 = <Response [404 Not Found]>.status_code

tests/test_main.py:58: AssertionError
=========================== short test summary info ============================
FAILED tests/test_main.py::test_verify_passed - assert 404 == 200
FAILED tests/test_main.py::test_verify_failed_needs_feedback - assert 404 == 200
FAILED tests/test_main.py::test_verify_max_iterations - assert 404 == 200
============================== 3 failed in 0.25s ===============================

```

## 2. Models Detected
- No pre-trained weights or models detected in this repository.
