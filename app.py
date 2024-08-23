import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Loading dataset
file_path = 'https://github.com/massa-atsu/osteoporosis/blob/main/osteoporosis.csv'
df = pd.read_csv(file_path)

# Data preprocessing
df = df.drop(columns=['Id', 'Alcohol Consumption', 'Medications'])
mode_value = df["Medical Conditions"].mode()[0]
df["Medical Conditions"].fillna(mode_value, inplace=True)
df.drop_duplicates(keep='first', inplace=True)

# Splitting data
target = 'Osteoporosis'
X = df.drop(columns=target)
y = df[target]

# One-Hot Encoding using pandas
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boost Model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Streamlit app
def main():
    st.title("Osteoporosis Prediction Web App")
    st.write("This app uses machine learning models to predict the likelihood of osteoporosis based on various factors.")

    st.sidebar.header("Input Features")
    
    def user_input_features():
        Age = st.sidebar.slider("Age", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].median()))
        Gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
        Calcium_Intake = st.sidebar.selectbox("Calcium Intake", df['Calcium Intake'].unique())
        Vitamin_D_Intake = st.sidebar.selectbox("Vitamin D Intake", df['Vitamin D Intake'].unique())
        Race_Ethnicity = st.sidebar.selectbox("Race/Ethnicity", df['Race/Ethnicity'].unique())
        Physical_Activity = st.sidebar.selectbox("Physical Activity", df['Physical Activity'].unique())
        Family_History = st.sidebar.selectbox("Family History", df['Family History'].unique())
        Prior_Fractures = st.sidebar.selectbox("Prior Fractures", df['Prior Fractures'].unique())
        Medical_Conditions = st.sidebar.selectbox("Medical Conditions", df['Medical Conditions'].unique())
        input_data = {
            'Age': Age,
            'Gender': Gender,
            'Calcium Intake': Calcium_Intake,
            'Vitamin D Intake': Vitamin_D_Intake,
            'Race/Ethnicity': Race_Ethnicity,
            'Physical Activity': Physical_Activity,
            'Family History': Family_History,
            'Prior Fractures': Prior_Fractures,
            'Medical Conditions': Medical_Conditions
        }
        return pd.DataFrame(input_data, index=[0])
    
    user_data = user_input_features()
    
    # One-Hot Encoding for user data
    user_data_encoded = pd.get_dummies(user_data)
    
    # Ensure the same columns as the training data
    user_data_encoded = user_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Prediction using Gradient Boosting
    gb_pred_proba = gb_model.predict_proba(user_data_encoded)[:, 1][0]
    gb_pred = gb_model.predict(user_data_encoded)[0]
    st.subheader("Gradient Boosting Model Prediction")
    st.write(f"Prediction: {'Osteoporosis' if gb_pred else 'No Osteoporosis'}")
    st.write(f"Probability of Osteoporosis: {gb_pred_proba:.2f}")

if __name__ == "__main__":
    main()
