# model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(out_path="iris_model.pkl"):#model path and training
    iris = load_iris()#loading dataset
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=150, random_state=42)#model
    clf.fit(X_train, y_train)

    joblib.dump(clf, out_path)#saving model
    print(f"Model trained and saved to {out_path}")

if __name__ == "__main__":
    train_model()
