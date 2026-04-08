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

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn 

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

## ✨ Features

- Interactive Streamlit UI for customer data input  
- Real-time churn prediction with probability score  
- Visual gauge chart for easy interpretation  
- Automatic preprocessing (encoding & scaling)  
- Handles missing/unseen data robustly  
- Supports trained model integration via session state  
- Demo fallback mode for quick testing  

## Notes For Model Integration

- To use a real trained model, load it into Streamlit session state as churn_model before running predictions.
- The current app is intentionally self-contained and includes a demo fallback score for quick testing.

  ## 📌 Use Case

This project can help banks identify customers at risk of leaving and take preventive actions such as targeted offers or engagement strategies.

## Future Improvements

- Add full preprocessing and feature engineering pipeline.
- Load trained model and preprocessors from serialized files.
- Add batch prediction via CSV upload.
- Add model explainability (SHAP or feature contribution view).
