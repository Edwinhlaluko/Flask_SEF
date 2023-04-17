import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import joblib as jb
import plotly.express as px

# Load the machine learning models
model1 = jb.load('linear_regression_model.joblib')
df = pd.read_csv('ESK2033.csv')
df = df.dropna()
#model2 = load('linear_regression_model.joblib')

# Define a function to make predictions using the models
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Define the layout of the app
st.set_page_config(page_title='Energy Dashboard App')
st.title('Energy Dashboard App')

# Add a sidebar
st.sidebar.title('Navigation')
menu = ['Home', 'Model 1', 'Model 2']
choice = st.sidebar.selectbox('Select a page', menu)

# Define the content of the pages
if choice == 'Home':
    st.header('Welcome to the Energy Dashboard App!')
    st.write('This app allows you to predict energy consumption based on different variables.')
    st.write('Please select a page from the sidebar to continue.')

elif choice == 'Model 1':
    st.header('Model 1')
    st.write('This model predicts energy consumption based on variables ResidualDemand, RSAContractedDemand, ThermalGeneration, PumpedWaterGeneration, and PumpedWaterSCOPumping.')
    with st.form(key='model1_form'):
        rsa_contracted_demand = st.number_input('RSA Contracted Demand', min_value=0, max_value=100, value=50, step=1)
        thermal_generation = st.number_input('Thermal Generation', min_value=0, max_value=100, value=50, step=1)
        pumped_water_generation = st.number_input('Pumped Water Generation', min_value=0, max_value=100, value=50, step=1)
        pumped_water_sco_pumping = st.number_input('Pumped Water SCO Pumping', min_value=0, max_value=100, value=50, step=1)
        submit_button = st.form_submit_button(label='Predict')
    if submit_button:
        input_data = [[rsa_contracted_demand, thermal_generation, pumped_water_generation, pumped_water_sco_pumping]]
        prediction = predict(model1, input_data)
        previous_demand = df['Residual Demand'].iloc[-2]
        percent_change = (prediction[0] - previous_demand) / previous_demand * 100
        st.success(f'Previous demand: {previous_demand:.2f}\nPredicted demand: {prediction[0]:.2f} ({percent_change:.2f}%)')

    fig = px.line(df, x=df.index, y='Residual Demand')
    
    st.plotly_chart(fig)

elif choice == 'Model 2':
    st.header('Model 2')
    st.write('This model predicts energy consumption based on variables Y1, Y2, Y3, and Y4.')
    with st.form(key='model2_form'):
        y1 = st.number_input('Y1', min_value=0, max_value=100, value=50, step=1)
        y2 = st.number_input('Y2', min_value=0, max_value=100, value=50, step=1)
        y3 = st.number_input('Y3', min_value=0, max_value=100, value=50, step=1)
        y4 = st.number_input('Y4', min_value=0, max_value=100, value=50, step=1)
        submit_button = st.form_submit_button(label='Predict')
    if submit_button:
        input_data = [[y1, y2, y3, y4]]
        prediction = predict(model2, input_data)
        st.success(f'The predicted energy consumption is {prediction[0]:.2f} units.')
# Add a footer
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('Developed by No-Name')
