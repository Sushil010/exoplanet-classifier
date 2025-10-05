import streamlit as st
import pandas as pd
import joblib
# from sklearn.externals import joblib

import xgboost as xgb
import matplotlib.pyplot as plt

# -------------------------
# Load saved pipeline and encoder
# -------------------------
best_model, feature_names = joblib.load("xgb_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")
class_names = list(label_encoder.classes_)

st.title("🌌 Exoplanet Classifier — Kepler/K2/TESS")

# -------------------------
# Sidebar: Hyperparameters (for reference)
# -------------------------
st.sidebar.header("⚙️ Model Hyperparameters")

n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200)
max_depth = st.sidebar.slider("max_depth", 1, 10, 6)
learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1)
subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8)
colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8)
gamma = st.sidebar.slider("gamma", 0, 10, 1)

# Build model object (visual reference)
model = xgb.XGBClassifier(
    n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
    subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
    use_label_encoder=False, eval_metric="logloss", random_state=42
)

# -------------------------
# Helper: Extract transformed feature names
# -------------------------
def get_transformed_feature_names(preprocessor):
    """Safely extract feature names after ColumnTransformer transformation."""
    names = []
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        pass

    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "named_steps"):
            last_step = list(trans.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                try:
                    out = last_step.get_feature_names_out(cols)
                    names.extend([f"{name}__{n}" for n in out])
                    continue
                except Exception:
                    pass
            names.extend([f"{name}__{c}" for c in cols])
        else:
            if hasattr(trans, "get_feature_names_out"):
                try:
                    out = trans.get_feature_names_out(cols)
                    names.extend([f"{name}__{n}" for n in out])
                    continue
                except Exception:
                    pass
            names.extend([f"{name}__{c}" for c in cols])
    return names

# -------------------------
# Input options
# -------------------------
st.header("📂 Input Data")

option = st.radio("Choose input method:", ["Upload File (CSV/Excel)", "Manual Entry"])

if option == "Upload File (CSV/Excel)":
    uploaded = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded:
        if uploaded.name.endswith(".csv"):
            user_df = pd.read_csv(uploaded)
        else:
            user_df = pd.read_excel(uploaded)
        st.write("Raw uploaded data:", user_df.head())

        try:
            X_input = user_df[feature_names]
        except KeyError:
            st.error("Uploaded file does not contain all required columns.")
            st.stop()

        if st.button("🔮 Predict from file"):
            preds = best_model.predict(X_input)
            probs = best_model.predict_proba(X_input)
            decoded_preds = label_encoder.inverse_transform(preds)

            results = pd.DataFrame({
                "Predicted_Label": decoded_preds,
                f"Prob_{class_names[0]}": probs[:, 0],
                f"Prob_{class_names[1]}": probs[:, 1]
            })
            st.write(results)

            # Download predictions
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv,
                               "exoplanet_predictions.csv", "text/csv")

            # Feature Importance chart
            st.subheader("🔭 Feature Importance")
            xgb_model = best_model.named_steps["model"]
            pre = best_model.named_steps["preprocessor"]

            # Get transformed feature names
            transformed_features = get_transformed_feature_names(pre)
            fi = xgb_model.feature_importances_

            # Align lengths defensively
            if len(fi) != len(transformed_features):
                st.warning(
                    f"Feature-name/importance length mismatch: "
                    f"{len(transformed_features)} names vs {len(fi)} importances. "
                    f"Aligning to the shorter length."
                )
                m = min(len(fi), len(transformed_features))
                fi = fi[:m]
                transformed_features = transformed_features[:m]

            importance = pd.Series(fi, index=transformed_features).sort_values(ascending=False).head(15)
            st.bar_chart(importance)

elif option == "Manual Entry":
    st.write("Enter values for a single observation:")
    input_data = {}
    for col in feature_names:
        val = st.text_input(f"Enter value for {col}")
        input_data[col] = [val]

    if st.button("🔮 Predict manually"):
        X_input = pd.DataFrame(input_data)[feature_names]
        preds = best_model.predict(X_input)
        probs = best_model.predict_proba(X_input)
        decoded_pred = label_encoder.inverse_transform(preds)[0]

        st.write(f"Prediction: **{decoded_pred}**")
        st.write(f"Probabilities → {class_names[0]}: {probs[0][0]:.3f}, {class_names[1]}: {probs[0][1]:.3f}")
