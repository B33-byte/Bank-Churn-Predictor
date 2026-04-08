import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Bank Churn Predictor", page_icon=":chart_with_upwards_trend:", layout="wide")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def get_demo_probability(features: dict) -> float:
    """Fallback score when a trained model is not available in session state."""
    score = (
        -2.2
        + 0.035 * max(features["age"] - 40, 0)
        + 0.22 * (0 if features["is_active_member"] else 1)
        + 0.30 * (1 if features["num_products"] <= 1 else 0)
        + 0.18 * min(features["tenure"] / 10.0, 1)
        + 0.25 * (1 if features["estimated_salary"] < 35000 else 0)
    )
    return float(sigmoid(score))


def get_model_probability(model, frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(frame)[0][1])

    pred = model.predict(frame)
    return float(pred[0])


def build_gauge(probability: float) -> go.Figure:
    value = probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 30], "color": "#d8f3dc"},
                    {"range": [30, 70], "color": "#ffe8a1"},
                    {"range": [70, 100], "color": "#ffccd5"},
                ],
                "threshold": {"line": {"color": "#d00000", "width": 3}, "value": 50},
            },
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=320)
    return fig


st.title("Bank Churn Predictor")
st.caption("Interactive Streamlit app to estimate customer churn probability.")

model = st.session_state.get("churn_model")
using_model = model is not None

if using_model:
    st.success("Loaded trained model from session state: churn_model")
else:
    st.info("No trained model found in session state. Using built-in demo scoring.")

with st.form("churn_form"):
    left, right = st.columns(2)

    with left:
        age = st.slider("Age", min_value=18, max_value=95, value=35)
        tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=3)
        balance = st.number_input("Account balance", min_value=0.0, value=50000.0, step=500.0)
        credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=650)

    with right:
        estimated_salary = st.number_input("Estimated salary", min_value=0.0, value=55000.0, step=500.0)
        num_products = st.slider("Number of products", min_value=1, max_value=4, value=2)
        has_credit_card = st.checkbox("Has credit card", value=True)
        is_active_member = st.checkbox("Is active member", value=True)

    submitted = st.form_submit_button("Predict churn")

if submitted:
    input_dict = {
        "credit_score": credit_score,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "num_products": num_products,
        "has_credit_card": int(has_credit_card),
        "is_active_member": int(is_active_member),
        "estimated_salary": estimated_salary,
    }
    input_df = pd.DataFrame([input_dict])

    if using_model:
        probability = get_model_probability(model, input_df)
    else:
        probability = get_demo_probability(input_dict)

    st.subheader("Prediction Result")
    if probability >= 0.5:
        st.error(f"High churn risk: {probability * 100:.1f}%")
    else:
        st.success(f"Low churn risk: {probability * 100:.1f}%")

    metric_col, gauge_col = st.columns([1, 2])
    with metric_col:
        st.metric(label="Churn probability", value=f"{probability * 100:.1f}%")
        st.write("Input snapshot")
        st.dataframe(input_df, use_container_width=True)

    with gauge_col:
        st.plotly_chart(build_gauge(probability), use_container_width=True)
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Churn Prediction", page_icon="🔮", layout="wide")

if 'churn_model' not in st.session_state or 'encoders' not in st.session_state or 'scaler' not in st.session_state:
    st.warning("Models and preprocessing objects not loaded. Please go to the main page to initialize.")
    st.stop()

model = st.session_state['churn_model']
encoders = st.session_state['encoders']
scaler = st.session_state['scaler']
df = st.session_state['data']

st.title("🔮 Interactive Churn Prediction")
st.markdown("Enter customer details to predict their likelihood of churning.")

with st.form("prediction_form"):
    st.subheader("Customer Information")
    col1, col2, col3 = st.columns(3)
    
    # Define features expected by the model in the exact order
    # The numerical cols scaled were: credit_sco, age, balance, monthly_ir, tenure_ye, nums_card, nums_service, last_transaction_month, engagement_score, risk_score
    # We must construct a DataFrame matching `df_ml` before scaling.
    
    with col1:
        credit_sco = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        gender = st.selectbox("Gender", options=df['gender'].dropna().unique().tolist() if 'gender' in df else ['male', 'female'])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        occupation = st.selectbox("Occupation", options=df['occupation'].dropna().unique().tolist() if 'occupation' in df else [])
    
    with col2:
        balance = st.number_input("Account Balance", min_value=0, value=50000000)
        monthly_ir = st.number_input("Monthly Income (VND)", min_value=0, value=15000000)
        tenure_ye = st.slider("Tenure (Years)", min_value=0, max_value=20, value=5)
        married = st.selectbox("Married Status (0=No, 1-3=Yes variants)", options=df['married'].dropna().unique().tolist() if 'married' in df else [0, 1, 2, 3])
        nums_card = st.number_input("Number of Cards", min_value=0, max_value=5, value=1)
        
    with col3:
        nums_service = st.number_input("Number of Services", min_value=1, max_value=10, value=2)
        active_member = st.checkbox("Active Member", value=True)
        last_transaction_month = st.number_input("Last Transaction (Months ago)", min_value=0, value=1)
        engagement_score = st.slider("Engagement Score", min_value=0, max_value=100, value=50)
        risk_score = st.number_input("Risk Score (e.g. 0.05)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    # Some categorical features might not be in the form, let's provide defaults for them based on simple modes
    origin_province = df['origin_province'].mode()[0] if 'origin_province' in df.columns else 'Unknown'
    customer_segment = df['customer_segment'].mode()[0] if 'customer_segment' in df.columns else 'Mass'
    loyalty_level = df['loyalty_level'].mode()[0] if 'loyalty_level' in df.columns else 'Bronze'
    digital_behavior = df['digital_behavior'].mode()[0] if 'digital_behavior' in df.columns else 'mobile'
    risk_segment = df['risk_segment'].mode()[0] if 'risk_segment' in df.columns else 'Low'
    cluster_group = df['cluster_group'].mode()[0] if 'cluster_group' in df.columns else 1

    submit_button = st.form_submit_button("Predict Churn Probability")

if submit_button:
    # Create an input dataframe matching original data before preprocessing
    input_data = {
        'credit_sco': [credit_sco],
        'gender': [gender],
        'age': [age],
        'occupation': [occupation],
        'balance': [balance],
        'monthly_ir': [monthly_ir],
        'origin_province': [origin_province],
        'tenure_ye': [tenure_ye],
        'married': [married],
        'nums_card': [nums_card],
        'nums_service': [nums_service],
        'active_member': [active_member],
        'last_transaction_month': [last_transaction_month],
        'customer_segment': [customer_segment],
        'engagement_score': [engagement_score],
        'loyalty_level': [loyalty_level],
        'digital_behavior': [digital_behavior],
        'risk_score': [risk_score],
        'risk_segment': [risk_segment],
        'cluster_group': [cluster_group]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Preprocess input data identically to the training set
    for col, enc in encoders.items():
        if col in input_df.columns:
            # Handle unseen labels by mapping to a known value or mode (simplified here)
            try:
                input_df[col] = enc.transform(input_df[col].astype(str))
            except ValueError:
                # If unknown category, just map to the first class (0)
                input_df[col] = 0
                
    for col in input_df.select_dtypes(include=['bool']).columns:
        input_df[col] = input_df[col].astype(int)
        
    numerical_cols = ['credit_sco', 'age', 'balance', 'monthly_ir', 'tenure_ye', 
                      'nums_card', 'nums_service', 'last_transaction_month', 
                      'engagement_score', 'risk_score']
    
    numerical_cols = [col for col in numerical_cols if col in input_df.columns]
    if numerical_cols:
         input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Make sure columns align with the model's expected features
    # Retrieve feature names from the model if possible (scikit-learn >= 1.0)
    try:
        model_features = model.feature_names_in_
        # Reorder and filter columns
        input_df = input_df[model_features]
    except AttributeError:
        pass # Older sklearn versions or models might not have this, rely on order matching

    # Predict
    probability = model.predict_proba(input_df)[0][1] # Probability of Class 'True'
    prediction = model.predict(input_df)[0]
    
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if probability > 0.5:
            st.error(f"### High Risk of Churn\nProbability: **{probability*100:.1f}%**")
        else:
             st.success(f"### Low Risk of Churn\nProbability: **{probability*100:.1f}%**")
             
    with col_res2:
        # Gauge chart using Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)