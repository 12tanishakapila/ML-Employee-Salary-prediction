import streamlit as st
import pandas as pd
import joblib

# Load model and scalers
model = joblib.load("salary_prediction_model.pkl")
age_scaler = joblib.load("age_scaler.pkl")
exp_scaler = joblib.load("exp_scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
degree_encoder = joblib.load("degree_encoder.pkl")
job_title_encoder = joblib.load("job_title_encoder.pkl")

st.set_page_config(page_title="Employee Salary Prediction", page_icon="🧑‍💼", layout="centered")

# Load dataset for dropdown options
data = pd.read_csv("Dataset09-Employee-salary-prediction.csv")
data.columns = ['Age', 'Gender', 'Degree', 'Job_Title', 'Experience_years', 'Salary']

st.markdown(
    "<h1 style='color:#ff5252; font-weight:bold;'>🧑‍💼 Welcome to the Employee Salary Prediction App!</h1>",
    unsafe_allow_html=True
)

st.markdown("""
This app provides **two ways to predict employee salaries** using a machine learning model trained on real-world data:

1. **Single Prediction:**  
   Enter employee details in the sidebar to instantly estimate their annual salary.

2. **Batch Prediction:**  
   Upload a CSV file with multiple employee records to get salary predictions for all at once.

**Features used for prediction:**
- Age
- Gender
- Degree
- Job Title
- Years of Experience

👉 For single prediction, fill in the sidebar and click **🎯 Predict Salary**. 
             
👉 For batch prediction, scroll down to the "Batch Prediction" section and upload your CSV file.
""")


st.sidebar.markdown(
    "### 📝 <span style='color:#ff5252;'>Enter Employee Details</span>",
    unsafe_allow_html=True
)

age = st.sidebar.slider("Age", min_value=int(data['Age'].min()), max_value=int(data['Age'].max()), value=int(data['Age'].mean()))
gender = st.sidebar.selectbox("Gender", options=data['Gender'].unique())
degree = st.sidebar.selectbox("Degree", options=data['Degree'].unique())
job_title = st.sidebar.selectbox("Job Title", options=data['Job_Title'].unique())
experience_years = st.sidebar.slider("Experience (years)", min_value=int(data['Experience_years'].min()), max_value=int(data['Experience_years'].max()), value=int(data['Experience_years'].mean()))

input_display = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Degree': degree,
    'Job Title': job_title,
    'Years of Experience': experience_years
}])

# Label encoding (must match training)
gender_encoded = gender_encoder.transform([gender])[0]
degree_encoded = degree_encoder.transform([degree])[0]
job_title_encoded = job_title_encoder.transform([job_title])[0]

# Scaling
age_scaled = age_scaler.transform([[age]])[0][0]
exp_scaled = exp_scaler.transform([[experience_years]])[0][0]

# Prepare input for prediction (must match training column order)
input_df = pd.DataFrame([{
    'Age_scaled': age_scaled,
    'Gender_Encode': gender_encoded,
    'Degree_Encode': degree_encoded,
    'Job_Title_Encode': job_title_encoded,
    'Experience_years_scaled': exp_scaled
}])

st.markdown("<h3 style='color:#ff5252;'>🔎 Input Data</h3>", unsafe_allow_html=True)
st.write(input_display)

# Predict button
if st.sidebar.button("Predict Salary"):
    st.balloons()
    prediction = model.predict(input_df)[0]
    st.markdown("---")
    st.markdown("<h2 style='color:#ff5252;'>💰 Salary Prediction Result</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background-color:#e8f5e9; padding:20px; border-radius:10px; text-align:center; margin-bottom:16px;">
            <span style="font-size:1.2em; color:#2e7d32;"><b>Predicted Salary:</b> {prediction:,.2f}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# Batch prediction section
st.markdown("---")
st.markdown("<h4 style='color:#ff5252;'>📂 Batch Prediction</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.markdown(
    "<span style='color:#ff5252; font-weight:bold;'>Uploaded data preview:</span>",
    unsafe_allow_html=True
    ) 
    st.write(batch_data.head())

    # Rename columns to match model expectations
    batch_data = batch_data.rename(columns={
        'Education Level': 'Degree',
        'Job Title': 'Job_Title',
        'Years of Experience': 'Experience_years'
    })


    # Drop rows with missing values in required columns
    required_cols = ['Age', 'Gender', 'Degree', 'Job_Title', 'Experience_years']
    before = len(batch_data)
    batch_data = batch_data.dropna(subset=required_cols)
    after = len(batch_data)
    if before != after:
       st.markdown(
        f"""
        <div style="background-color:#fff3cd; border-left:6px solid #b71c1c; color:#222; padding:16px; border-radius:6px; margin-bottom:16px;">
            <span style='color:#b71c1c; font-weight:bold;'>Notice:</span><br>
            {before - after} row(s) with missing or invalid values were removed and will not be included in the batch prediction results.
        </div>
        """,
        unsafe_allow_html=True
        )

    # Preprocess batch data (encoding & scaling)
    batch_data['Gender_Encode'] = gender_encoder.transform(batch_data['Gender'])
    batch_data['Degree_Encode'] = degree_encoder.transform(batch_data['Degree'])
    batch_data['Job_Title_Encode'] = job_title_encoder.transform(batch_data['Job_Title'])
    batch_data['Age_scaled'] = age_scaler.transform(batch_data[['Age']])
    batch_data['Experience_years_scaled'] = exp_scaler.transform(batch_data[['Experience_years']])

    # Prepare input columns (must match model training order)
    input_cols = ['Age_scaled', 'Gender_Encode', 'Degree_Encode', 'Job_Title_Encode', 'Experience_years_scaled']
    batch_preds = model.predict(batch_data[input_cols])
    batch_data['PredictedSalary'] = batch_preds

    

    st.markdown(
    """
    <div style="background-color:#e8f5e9; padding:20px; border-radius:10px; text-align:center; margin-bottom:16px;">
        <span style="font-size:1.2em; color:#2e7d32;"><b>✅ Batch predictions completed!</b></span>
    </div>
    """,
    unsafe_allow_html=True
    ) 

    # Rename columns for display
    display_df = batch_data.rename(columns={
        'Degree': 'Education Level',
        'Job_Title': 'Job Title',
        'Experience_years': 'Years of Experience'
    })

    st.markdown("<h3 style='color:#ff5252;'>📊 Batch Prediction Results</h3>", unsafe_allow_html=True)

    st.write(display_df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'PredictedSalary']].head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Predictions CSV",
        csv,
        file_name='predicted_salaries.csv',
        mime='text/csv'
    )
