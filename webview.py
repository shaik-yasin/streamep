import streamlit as st
import numpy as np
import pandas as pd 
import joblib

st.title("promotion prediction App")

df = pd.read_csv('train_LZdllcl.csv')

department = st.selectbox("Department",pd.unique((df['department'])))
region = st.selectbox("region",pd.unique(df['region']))
education = st.selectbox("Education",pd.unique(df['education']))
gender = st.selectbox("Gender",pd.unique(df['gender']))
recruitment_channel = st.selectbox("Recruitment_channel",pd.unique(df['recruitment_channel']))

no_of_trainings = st.number_input("No_of_training")
age = st.number_input("age")
previous_year_rating = st.number_input("previous year rating")
length_of_service = st.number_input("length of service")
KPIs_met_80 = st.number_input("KPIs_met >80%")
awards_won = st.number_input("Entre Awards_won?")
avg_training_score = st.number_input("Avg_training_score")

input = {
'department': department,
'region': region,
'education': education,
'gender': gender,
'recruitment_channel': recruitment_channel,
'no_of_trainings': no_of_trainings,
'age': age,
'previous_year_rating': previous_year_rating,
'length_of_service': length_of_service,
'KPIs_met >80%': KPIs_met_80,
'awards_won?': awards_won,
'avg_training_score': avg_training_score

}

model = joblib.load('promote_pipeline_model.pkl')

if st.button('Predict'):
    X_input = pd.DataFrame(input,index=[0])
    prediction = model.predict(X_input)
    st.write("The predicted value is:")
    st.write(prediction)

# 'previous_year_rating'
# 'length_of_service'
# 'KPIs_met >80%'
# 'awards_won?'
# 'avg_training_score'
