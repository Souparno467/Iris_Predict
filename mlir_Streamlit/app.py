import streamlit as st
import numpy as np
import pandas as pd
from services import (
    df, predict, plot_scatter, plot_histogram, plot_pie,
    plot_3d_scatter, plot_bar_avg, plot_probability_bar,
    heatmap_fig, pairplot_fig, dataset_csv_bytes, model_bytes, get_model
)

# Page config
st.set_page_config(
    page_title="Iris ML Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme dark/light
theme = st.sidebar.radio("Theme", ("Dark", "Light"), index=0)

dark_css = """
<style>
    .stApp { background: linear-gradient(135deg,#0f172a,#001219); color: #ffffff; }
    .block-container{background-color: rgba(255,255,255,0.03); border-radius:12px; padding:18px;}
    h1, h2, h3, .big-font { color: #f8fafc; }
    .section-title { font-weight:700; color:#e2e8f0; }
</style>
"""

light_css = """
<style>
    .stApp { background: linear-gradient(135deg,#f8fafc,#eef2ff); color: #0f172a; }
    .block-container{background-color: rgba(0,0,0,0.02); border-radius:12px; padding:18px;}
    h1, h2, h3, .big-font { color: #0f172a; }
    .section-title { font-weight:700; color:#0b1220; }
</style>
"""

scroll_css = """
<style>
    .main { overflow-y: auto; max-height: 100vh; }
</style>
"""

st.markdown(dark_css if theme == "Dark" else light_css, unsafe_allow_html=True)
st.markdown(scroll_css, unsafe_allow_html=True)

# Header
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Iris Machine Learning Dashboard")
    st.markdown("**Interactive Streamlit dashboard** — prediction + exploration + download.")
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_versicolor_3.jpg", width=110)

st.markdown("---")

# Sidebar inputs
st.sidebar.header("Input / Controls")
input_mode = st.sidebar.selectbox("Input mode", ["Sliders (recommended)", "Manual text inputs"])

if input_mode == "Sliders (recommended)":
    sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()), float(df["sepal length (cm)"].median()))
    sepal_width  = st.sidebar.slider("Sepal Width  (cm)", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()), float(df["sepal width (cm)"].median()))
    petal_length = st.sidebar.slider("Petal Length (cm)", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()), float(df["petal length (cm)"].median()))
    petal_width  = st.sidebar.slider("Petal Width  (cm)", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()), float(df["petal width (cm)"].median()))
else:
    sepal_length = st.sidebar.text_input("Sepal Length (cm)", placeholder="e.g. 5.1")
    sepal_width  = st.sidebar.text_input("Sepal Width (cm)",  placeholder="e.g. 3.5")
    petal_length = st.sidebar.text_input("Petal Length (cm)", placeholder="e.g. 1.4")
    petal_width  = st.sidebar.text_input("Petal Width (cm)", placeholder="e.g. 0.2")

# Utilities
st.sidebar.markdown("### Utilities")
st.sidebar.download_button("Download dataset (CSV)", data=dataset_csv_bytes(), file_name="iris_dataset.csv", mime="text/csv")
try:
    model_bin = model_bytes()
    st.sidebar.download_button("Download model (joblib)", data=model_bin, file_name="iris_model.pkl", mime="application/octet-stream")
except Exception:
    st.sidebar.write("Model not found. Run `python model.py` first.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Placeholders**")
st.sidebar.write("- Model explainability (SHAP) placeholder")
st.sidebar.write("- Advanced settings placeholder")

# Quick Prediction
st.markdown("### Quick Prediction")
quick_col1, quick_col2, quick_col3 = st.columns(3)

with quick_col1:
    st.write("Enter features and click Predict")
    st.write(f"**Sepal L**: {sepal_length}")
    st.write(f"**Sepal W**: {sepal_width}")

with quick_col2:
    st.write(f"**Petal L**: {petal_length}")
    st.write(f"**Petal W**: {petal_width}")

with quick_col3:
    if st.button("Predict", use_container_width=True):
        predict_button = True
    else:
        predict_button = False
    prediction_placeholder = st.empty()

# Tabs
tabs = st.tabs(["Prediction", "Dataset", "Visualizations", "Correlations", "Model & Info"])

with tabs[0]:
    st.header(" Prediction")
    st.markdown("Use the inputs from the sidebar. Prediction uses the trained RandomForest model and returns probabilities for all classes.")

    if predict_button:
        try:
            features = [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]
        except Exception:
            prediction_placeholder.error("Invalid input values. Use numeric values.")
            features = None

        if features:
            try:
                result = predict(features)
                pred_name = result["class_name"]
                proba_array = result["proba_array"]

                # Display predicted class clearly
                species_emojis = {
                    "setosa": "🌱",
                    "versicolor": "🌸",
                    "virginica": "🌺"
                }
                emoji = species_emojis.get(pred_name.lower(), "")
                prediction_placeholder.subheader("🌼 Prediction Result")
                prediction_placeholder.success(f"**Predicted Iris species:** {emoji} `{pred_name}`")

                prediction_placeholder.info("Raw probabilities are shown below.")

                prob_df = pd.DataFrame({
                    "class": list(result["probabilities"].keys()),
                    "probability": list(result["probabilities"].values())
                }).sort_values("probability", ascending=False)
                prediction_placeholder.table(prob_df.style.format({"probability": "{:.3f}"}))

                fig_probs = plot_probability_bar(proba_array)
                prediction_placeholder.plotly_chart(fig_probs, use_container_width=True)

                prediction_placeholder.markdown("**Explainability:** placeholder for SHAP/feature contributions (add later).")

            except FileNotFoundError as fe:
                prediction_placeholder.error(str(fe))
            except Exception as e:
                prediction_placeholder.error(f"Prediction failed: {e}")
    else:
        st.write("Click **Predict** to get a prediction based on the inputs.")

with tabs[1]:
    st.header("📊 Dataset")
    st.markdown("Preview of the Iris dataset. Use the controls to download the CSV.")
    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("**Summary statistics**")
    st.dataframe(df.describe().T)

with tabs[2]:
    st.header("📈 Visualizations")
    st.markdown("Interactive Plotly charts for exploration.")
    vcol1, vcol2 = st.columns([2, 3])

    with vcol1:
        st.subheader("Scatter (interactive)")
        st.plotly_chart(plot_scatter(), use_container_width=True)

        st.subheader("Histogram")
        feature_choice = st.selectbox("Feature (histogram)", df.columns[:-1].tolist(), index=0)
        st.plotly_chart(plot_histogram(feature_choice), use_container_width=True)

    with vcol2:
        st.subheader("3D Scatter")
        st.plotly_chart(plot_3d_scatter(), use_container_width=True)

        st.subheader("Average feature by species")
        st.plotly_chart(plot_bar_avg("petal length (cm)"), use_container_width=True)

with tabs[3]:
    st.header(" Correlation & Pairplot")
    st.markdown("Seaborn heatmap and pairplot for deeper statistical inspection.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Correlation heatmap")
        fig_hm = heatmap_fig()
        st.pyplot(fig_hm)

    with col_b:
        with st.expander(" Pairplot (may take a few seconds)", expanded=False):
            with st.spinner("Rendering pairplot..."):
                fig_pp = pairplot_fig()
                st.pyplot(fig_pp)

with tabs[4]:
    st.header(" Model & App Info")
    st.markdown("Information about the model, training, and placeholders for advanced features.")

    try:
        model = get_model()
        st.markdown("**Model loaded successfully.**")
        st.write(model)

        with st.expander(" Model Parameters", expanded=False):
            st.json({k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool))})

    except Exception as e:
        st.warning("Model not loaded. Run `python model.py` to train and save the model first.")
        st.write(e)

    st.markdown("**Placeholders**")
    st.write("- Explainability (SHAP) integration")
    st.write("- CI/CD and tests placeholder")
    st.write("- Multiple model versions / A/B testing placeholder")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Model: RandomForest • Data: Iris dataset")
