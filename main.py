import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import joblib as jb
import plotly.express as px
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

st.set_option('deprecation.showPyplotGlobalUse', False)


# Load the machine learning models
model = jb.load('ann_weather.joblib')
model1 = jb.load('linear_regression_model.joblib')
df = pd.read_csv('ESK2033.csv')
df = df.dropna()
#model2 = load('linear_regression_model.joblib')

# Define a function to make predictions using the models
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Define the layout of the app
st.set_page_config(page_title='SmartEnergyForecast(SEF) Dashboard App')
st.title('SmartEnergyForecast(SEF) Dashboard App')

# Add a sidebar
st.sidebar.title('Navigation')
menu = ['Home', 'Prediction Model', 'Clustering Model']
choice = st.sidebar.selectbox('Select a page', menu)

# Define the content of the pages
if choice == 'Home':
    st.header('Welcome to the SmartEnergyForecast(SEF) Dashboard App!')
    st.write('This app allows us to predict energy demand based on different variables, It also allows us to make recommendations of renewable energy source based on weather patterns of a specific weather station.')
    st.write('Please select a page from the sidebar to continue.')

elif choice == 'Prediction Model':
    st.header('Prediction Model')
    st.write('This model predicts energy consumption based on variables ResidualDemand, RSAContractedDemand, ThermalGeneration, PumpedWaterGeneration, and PumpedWaterSCOPumping.')
    with st.form(key='model1_form'):
        rsa_contracted_demand = st.number_input('RSA Contracted Demand', min_value=0, max_value=100000, value=50, step=1)
        thermal_generation = st.number_input('Thermal Generation', min_value=0, max_value=100000, value=50, step=1)
        pumped_water_generation = st.number_input('Pumped Water Generation', min_value=0, max_value=100000, value=50, step=1)
        pumped_water_sco_pumping = st.number_input('Pumped Water SCO Pumping', min_value=0, max_value=100000, value=50, step=1)
        submit_button = st.form_submit_button(label='Predict')
    if submit_button:
        input_data = [[rsa_contracted_demand, thermal_generation, pumped_water_generation, pumped_water_sco_pumping]]
        prediction = predict(model1, input_data)
        previous_demand = df['Residual Demand'].iloc[-2]
        percent_change = (prediction[0] - previous_demand) / previous_demand * 100
        st.success(f'Previous demand: {previous_demand:.2f}\nPredicted demand: {prediction[0]:.2f} ({percent_change:.2f}%)')

    fig = px.line(df, x=df.index, y='Residual Demand')
    
    st.plotly_chart(fig)

elif choice == 'Clustering Model':
    st.header('Clustering Model')
    st.write('This model predicts energy consumption based on variables Y1, Y2, Y3, and Y4.')
        
    # Get the selected season from the user
    season = st.selectbox('Select a season', ['winter', 'spring', 'summer', 'fall'])
    data = pd.read_csv('station_weather_data.csv')
    data = data.dropna()


    # Filter the data by the selected seasom

    seasons = {
    12: 'winter',
    1: 'winter',
    2: 'winter',
    3: 'spring',
    4: 'spring',
    5: 'spring',
    6: 'summer',
    7: 'summer',
    8: 'summer',
    9: 'fall',
    10: 'fall',
    11: 'fall'
    }
    data['DATE'] = pd.to_datetime(data['DATE'])
    # Add a new column for the season
    data['SEASON'] = data['DATE'].apply(lambda x: seasons[x.month])
    # Load the CSV data
    
    season_data = data[data['SEASON'] == season]
    
  
m = folium.Map(location=[data['LATITUDE'].mean(), data['LONGITUDE'].mean()], zoom_start=10)
marker_cluster = MarkerCluster().add_to(m)

for index, row in data.iterrows():
    prcp = row['PRCP']
    tmax = row['TMAX']
    tmin = row['TMIN']
    tavg = row['TAVG']
    input_data = [[prcp, tmax, tmin, tavg]]
    energy_type = model.predict(input_data)[0]
    
    if energy_type == 'Solar':
        color = 'green'
    elif energy_type == 'Non-Renewable':
        color = 'red'
    else:
        color = 'gray'
        
    folium.Marker([row['LATITUDE'], row['LONGITUDE']], 
                  popup=f"Station: {row['STATION']}\nEnergy Type: {energy_type}", 
                  icon=folium.Icon(color=color)).add_to(marker_cluster)
    
    # Add location icons with different colors
    if energy_type == 'Solar':
        icon_color = 'green'
    elif energy_type == 'Wind':
        icon_color = 'red'
    else:
        icon_color = 'gray'
        
    folium.Marker([row['LATITUDE'], row['LONGITUDE']], 
                  icon=folium.Icon(color=icon_color)).add_to(m)

folium_static(m)



    # # Plot the data on a map
    # m = folium.Map(location=[data['LATITUDE'].mean(), data['LONGITUDE'].mean()], zoom_start=10)
    # for index, row in data.iterrows():
    #     prcp = row['PRCP']
    #     tmax = row['TMAX']
    #     tmin = row['TMIN']
    #     tavg = row['TAVG']
    #     input_data = [[prcp, tmax, tmin, tavg]]
    #     energy_type = model.predict(input_data)[0]
    #     folium.Marker([row['LATITUDE'], row['LONGITUDE']], popup=f"Station: {row['STATION']}\nEnergy Type: {energy_type}").add_to(m)
    # folium_static(m)
    


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
