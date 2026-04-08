# Bank Churn Predictor

This project is a Streamlit web app for customer churn prediction.

It provides an interactive form where users enter customer profile details, and the app returns churn risk as a percentage with a visual gauge.

## How Streamlit Is Used

- Streamlit builds the full user interface in Python.
- The input form is created with Streamlit widgets like sliders, checkboxes, and number inputs.
- Prediction results are shown using Streamlit messages and metrics.
- A Plotly gauge chart is embedded in Streamlit to visualize churn probability.
- If a trained model is available in Streamlit session state as churn_model, the app uses it.
- If no model is loaded, the app uses a built-in demo scoring function so the interface still works.

## Project Structure

- main.py: Streamlit application entry point

## Requirements

- Python 3.9+
- streamlit
- pandas
- numpy
- plotly

## Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install streamlit pandas numpy plotly
```

## Run The App

Start the app with:

```bash
streamlit run main.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Notes For Model Integration

- To use a real trained model, load it into Streamlit session state as churn_model before running predictions.
- The current app is intentionally self-contained and includes a demo fallback score for quick testing.

## Future Improvements

- Add full preprocessing and feature engineering pipeline.
- Load trained model and preprocessors from serialized files.
- Add batch prediction via CSV upload.
- Add model explainability (SHAP or feature contribution view).
