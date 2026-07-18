from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Redshifted",
    page_icon="🌌",
    layout="wide",
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

CLASSIFIER_PATH = MODELS_DIR / "object_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "redshift_regressor_deploy.joblib"


# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------

@st.cache_resource
def load_model_artifacts():
    """Load the classification and deployment regression artifacts."""

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier artifact was not found: {CLASSIFIER_PATH}"
        )

    if not REGRESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Regressor artifact was not found: {REGRESSOR_PATH}"
        )

    classifier_artifact = joblib.load(CLASSIFIER_PATH)
    regressor_artifact = joblib.load(REGRESSOR_PATH)

    return classifier_artifact, regressor_artifact


try:
    classifier_artifact, regressor_artifact = load_model_artifacts()

except FileNotFoundError as error:
    st.error(str(error))
    st.info(
        "Run the model-export cells in notebooks 04 and 05 before "
        "starting the Streamlit application."
    )
    st.stop()

except Exception as error:
    st.error(f"Failed to load the model artifacts: {error}")
    st.stop()


classifier = classifier_artifact["model"]
regressor = regressor_artifact["model"]

classifier_features = classifier_artifact["features"]
regressor_features = regressor_artifact["features"]


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def create_photometric_features(
    dered_u: float,
    dered_g: float,
    dered_r: float,
    dered_i: float,
    dered_z: float,
) -> pd.DataFrame:
    """Create raw-magnitude and color-index features."""

    feature_values = {
        "dered_u": dered_u,
        "dered_g": dered_g,
        "dered_r": dered_r,
        "dered_i": dered_i,
        "dered_z": dered_z,
        "u_g": dered_u - dered_g,
        "g_r": dered_g - dered_r,
        "r_i": dered_r - dered_i,
        "i_z": dered_i - dered_z,
    }

    return pd.DataFrame([feature_values])


# ---------------------------------------------------------
# Page content
# ---------------------------------------------------------

st.title("🌌 Redshifted")

st.write(
    "Classify an SDSS celestial object as a STAR, GALAXY, or QSO "
    "using dereddened photometric magnitudes."
)

st.write(
    "For objects classified as GALAXY, the app also estimates "
    "their redshift."
)


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------

st.subheader("Photometric Input")

st.caption(
    "Enter the five dereddened SDSS magnitudes. "
    "Color indices are calculated automatically."
)

with st.form("photometric_input_form"):
    column_1, column_2 = st.columns(2)

    with column_1:
        dered_u = st.number_input(
            "dered_u",
            value=20.00,
            step=0.01,
            format="%.4f",
        )

        dered_g = st.number_input(
            "dered_g",
            value=19.20,
            step=0.01,
            format="%.4f",
        )

        dered_r = st.number_input(
            "dered_r",
            value=18.70,
            step=0.01,
            format="%.4f",
        )

    with column_2:
        dered_i = st.number_input(
            "dered_i",
            value=18.40,
            step=0.01,
            format="%.4f",
        )

        dered_z = st.number_input(
            "dered_z",
            value=18.20,
            step=0.01,
            format="%.4f",
        )

    submitted = st.form_submit_button(
        "Analyze object",
        type="primary",
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:
    input_df = create_photometric_features(
        dered_u=dered_u,
        dered_g=dered_g,
        dered_r=dered_r,
        dered_i=dered_i,
        dered_z=dered_z,
    )

    classifier_input = input_df[classifier_features]

    predicted_class = classifier.predict(classifier_input)[0]
    probabilities = classifier.predict_proba(classifier_input)[0]

    probability_df = pd.DataFrame({
        "class": classifier.classes_,
        "probability": probabilities,
    }).sort_values(
        "probability",
        ascending=False,
    )

    predicted_probability = float(
        probability_df.loc[
            probability_df["class"] == predicted_class,
            "probability",
        ].iloc[0]
    )

    st.divider()
    st.subheader("Prediction Result")

    result_column_1, result_column_2 = st.columns(2)

    with result_column_1:
        st.metric(
            "Predicted object class",
            predicted_class,
        )

    with result_column_2:
        st.metric(
            "Classification confidence",
            f"{predicted_probability:.2%}",
        )

    st.write("### Class probabilities")

    display_probability_df = probability_df.copy()
    display_probability_df["probability"] = (
        display_probability_df["probability"]
        .map(lambda value: f"{value:.2%}")
    )

    st.dataframe(
        display_probability_df,
        hide_index=True,
    )

    st.write("### Derived color indices")

    color_columns = ["u_g", "g_r", "r_i", "i_z"]

    st.dataframe(
        input_df[color_columns].round(4),
        hide_index=True,
    )

    if predicted_class == "GALAXY":
        regressor_input = input_df[regressor_features]
        predicted_redshift = float(
            regressor.predict(regressor_input)[0]
        )

        st.success(
            "This object was classified as GALAXY, "
            "so redshift estimation is available."
        )

        st.metric(
            "Estimated galaxy redshift (z)",
            f"{predicted_redshift:.4f}",
        )

        st.caption(
            "Live redshift prediction uses the compact Gradient "
            "Boosting deployment model."
        )

    else:
        st.info(
            "Redshift estimation is only produced for objects "
            "classified as GALAXY."
        )


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

st.divider()
st.subheader("Models used")

st.write(
    f"**Object classification:** "
    f"{classifier_artifact['model_name']}"
)

st.write(
    f"**Live redshift prediction:** "
    f"{regressor_artifact['model_name']}"
)