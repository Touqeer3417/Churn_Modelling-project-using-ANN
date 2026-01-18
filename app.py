import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle
import streamlit as st
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('churn_model.h5')

# Load the label encoder
with open('label_Encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the scaler
with open('scalar.pkl', 'rb') as f:
    scalar = pickle.load(f)

# Load the one-hot encoder
with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)


## Streamlit App Code Here ##

st.title("Customer Churn Prediction")
# Input fields
CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
Geography = st.selectbox("Geography", options=['France', 'Spain', 'Germany'])
Gender = st.selectbox("Gender", options=['Male', 'Female'])
Age = st.slider("Age", 18,92)
Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
Balance = st.number_input("Balance", min_value=0.0, value=1000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card", options=[0, 1])
IsActiveMember = st.selectbox("Is Active Member", options=[0, 1])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
# Create a dataframe for the input
input_data = {
    'CreditScore': [CreditScore],
    'Geography': [Geography],
    'Gender': [Gender],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember], 
    'EstimatedSalary': [EstimatedSalary]
}
input_df = pd.DataFrame(input_data)
# Encode Geography using OneHotEncoder
geo_encoded = onehot_encoder.transform([input_data['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))


#encode gender
input_df['Gender'] = label_encoder.transform(input_df['Gender'])    
#drop geography column
input_df = input_df.drop('Geography', axis=1)

# Combine the encoded geography columns with the original input dataframe
input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

input_df_scaled = scalar.transform(input_df)

result = model.predict(input_df_scaled)[0][0]

st.write(f"Churn Probability: {result:.2f}")

if result > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")