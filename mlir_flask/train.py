from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "iris_rf.joblib"#path to save the model

def main():
    data = load_iris()#load the iris dataset
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # to maintain class distribution
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(
        {
            "estimator": clf, # the trained model
            "target_names": data.target_names,# the class names
            "feature_names": data.feature_names,# the feature names
        },
        MODEL_PATH,#path to save the model
    )
    print(f"Saved model to {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    main()
