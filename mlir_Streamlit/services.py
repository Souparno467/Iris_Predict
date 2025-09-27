# services.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Paths
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "iris_model.pkl"

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Lazy-load model
_model = None
def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run `python model.py` first.")
        _model = joblib.load(MODEL_PATH)
    return _model

# Prediction function 
def predict(features):
    """
    features: list or array-like of 4 floats [sepal_length, sepal_width, petal_length, petal_width]
    returns: dict with class_name, class_index, probabilities (dict) and proba_array (numpy array)
    """
    model = get_model()
    arr = np.array(features, dtype=float).reshape(1, -1)
    pred_idx = int(model.predict(arr)[0])
    proba = model.predict_proba(arr)[0]
    return {
        "class_index": pred_idx,
        "class_name": str(iris.target_names[pred_idx]),
        "probabilities": {str(name): float(p) for name, p in zip(iris.target_names, proba)},
        "proba_array": proba
    }

# Plotly plots (interactive)
def plot_scatter():
    return px.scatter(df, x="sepal length (cm)", y="sepal width (cm)",
                      color="species", title="Sepal Length vs Sepal Width",
                      labels={"sepal length (cm)": "Sepal Length (cm)", "sepal width (cm)": "Sepal Width (cm)"})

def plot_histogram(feature="sepal length (cm)"):
    return px.histogram(df, x=feature, color="species", barmode="overlay",
                        title=f"Distribution of {feature}")

def plot_pie():
    return px.pie(df, names="species", title="Species Distribution")

def plot_3d_scatter():
    return px.scatter_3d(df, x="sepal length (cm)", y="sepal width (cm)",
                         z="petal length (cm)", color="species", symbol="species",
                         title="3D Scatterplot")

def plot_bar_avg(feature="petal length (cm)"):
    avg = df.groupby("species").mean().reset_index()
    return px.bar(avg, x="species", y=feature, color="species", title=f"Average {feature} by Species")

def plot_probability_bar(proba_array):
    prob_df = pd.DataFrame({
        "class": iris.target_names,
        "probability": proba_array
    })
    fig = px.bar(prob_df, x="class", y="probability", color="class",
                 range_y=[0, 1], title="Prediction Probabilities")
    fig.update_traces(texttemplate="%{y:.2f}", textposition="outside")
    return fig

# Seaborn/Matplotlib  plots
def heatmap_fig():
    corr = df.drop(columns=["species"]).corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    fig.tight_layout()
    return fig

def pairplot_fig():
  
    pair = sns.pairplot(df, hue="species", diag_kind="kde", palette="husl", corner=False)
    fig = pair.fig
    fig.tight_layout()
    return fig

# Utilities
def dataset_csv_bytes():
    return df.to_csv(index=False).encode("utf-8")

def model_bytes():
    # returns bytes of the saved model file for download
    with open(MODEL_PATH, "rb") as f:
        return f.read()
