# Employee Salary Prediction App

This project is a web application built with Streamlit that predicts employee salaries based on user input such as age, gender, degree, job title, and years of experience. The app uses a machine learning model trained on a dataset of employee salaries.

## Features

- User-friendly web interface with Streamlit
- Predicts salary based on input features
- Uses a pre-trained machine learning model
- Scales numerical features and encodes categorical variables

## Requirements

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## Files

- `salary_prediction_app.py` &mdash; Main Streamlit application
- `salary_prediction_model.pkl` &mdash; Trained machine learning model
- `Dataset09-Employee-salary-prediction.csv` &mdash; Dataset used for training and encoding
- `requirements.txt` &mdash; List of required Python packages
- `age_scaler.pkl` &mdash; Scaler for the Age feature
- `exp_scaler.pkl` &mdash; Scaler for the Experience_years feature
- `degree_encoder.pkl` &mdash; Encoder for the Degree feature
- `gender_encoder.pkl` &mdash; Encoder for the Gender feature
- `job_title_encoder.pkl` &mdash; Encoder for the Job Title feature

## Usage

### 1. Prepare the Files

Ensure the following files are in your project directory:
- `salary_prediction_app.py`
- `salary_prediction_model.pkl`
- `Dataset09-Employee-salary-prediction.csv`
- `age_scaler.pkl`
- `exp_scaler.pkl`
- `degree_encoder.pkl`
- `gender_encoder.pkl`
- `job_title_encoder.pkl`

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Start the Streamlit app with:

```bash
streamlit run salary_prediction_app.py
```

### 4. Use the App

- Enter the required details: age, gender, degree, job title, and years of experience.
- Click the **Predict Salary** button to see the predicted salary.

## Notes

- Ensure that the paths to `salary_prediction_model.pkl` and `Dataset09-Employee-salary-prediction.csv` are correct in your code.
- The application uses `StandardScaler` to scale the 'Age' and 'Experience_years' features before making predictions.
- Categorical variables are encoded based on their unique values in the dataset. Adjust the encoding logic if your dataset changes.

---

**Enjoy predicting employee salaries!**