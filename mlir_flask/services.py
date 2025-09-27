from pathlib import Path
import joblib
import numpy as np

_MODEL = None # This will store our loaded model bundle

def get_model_bundle():
    global _MODEL
    if _MODEL is None:
        model_path = Path(__file__).resolve().parent / "model" / "iris_rf.joblib"
        _MODEL = joblib.load(model_path)
    return _MODEL

def predict_iris(features):
    bundle = get_model_bundle()
    clf = bundle["estimator"] 
    target_names = bundle["target_names"]# e.g., ['setosa', 'versicolor', 'virginica']
    X = np.array([features], dtype=float)
    proba = clf.predict_proba(X)[0]# e.g., [0.1, 0.7, 0.2]
    idx = int(np.argmax(proba))
    return {
        "class_index": idx,
        "class_name": str(target_names[idx]),
        "probabilities": {str(name): float(p) for name, p in zip(target_names, proba)},
    }
