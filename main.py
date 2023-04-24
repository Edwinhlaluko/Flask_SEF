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
from PIL import Image
import os
import plotly.graph_objs as go


st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the machine learning models
model = jb.load('ann_weather.joblib')
model1 = jb.load('linear_regression_model.joblib')
df = pd.read_csv('ESK2033.csv')
df = df.dropna()

# Define a function to make predictions using the models
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Define the layout of the app
st.set_page_config(page_title='SmartEnergyForecast(SEF) Dashboard App')
st.title('SmartEnergyForecast(SEF) Dashboard App')

# Add a sidebar
st.sidebar.title('Navigation')
menu = ['Home', 'Prediction Model', 'Clustering Model', 'Dashboard']
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
        rsa_contracted_demand = st.number_input('RSA Contracted Demand', min_value=-100000, max_value=100000, value=50, step=1)
        thermal_generation = st.number_input('Thermal Generation', min_value=-100000, max_value=100000, value=50, step=1)
        pumped_water_generation = st.number_input('Pumped Water Generation', min_value=-100000, max_value=100000, value=50, step=1)
        pumped_water_sco_pumping = st.number_input('Pumped Water SCO Pumping', min_value=-100000, max_value=100000, value=50, step=1)
        submit_button = st.form_submit_button(label='Predict')
    if submit_button:
        input_data = [[rsa_contracted_demand, thermal_generation, pumped_water_generation, pumped_water_sco_pumping]]
        prediction = predict(model1, input_data)
        previous_demand = df['Residual Demand'].iloc[-2]
        percent_change = (prediction[0] - previous_demand) / previous_demand * 100
        st.success(f'Previous demand: {previous_demand:.2f}\nPredicted demand: {prediction[0]:.2f} ({percent_change:.2f}%)')

elif choice == 'Clustering Model':
    st.header('Clustering Model')
    st.write('This page shows recommendations of renewable energy source based on weather patterns of a specific weather station.')
        
    # Get the selected season from the user
    season = st.selectbox('Select a season', ['winter', 'spring', 'summer', 'fall'])
    data = pd.read_csv('station_weather_data.csv')
    data = data.dropna()

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

elif choice == 'Dashboard':  # Add new elif statement
    st.header('SmartEnergyForecast(SEF) Dashboard')
    st.write('This page shows recommendations of renewable energy source based on weather patterns of a specific weather station.')
    data_eskom=pd.read_csv("ESK2033.csv")
    gff=pd.read_csv("myupdateddata.csv")
  
    df['Timestamp'] = pd.to_datetime(df['Date Time Hour Beginning'])

    #Get units of time from the timestamp
    df['min'] = df['Timestamp'].dt.minute
    df['hour'] = df['Timestamp'].dt.hour
    df['wday'] = df['Timestamp'].dt.dayofweek
    df['mday'] = df['Timestamp'].dt.day
    df['yday'] = df['Timestamp'].dt.dayofyear
    df['month'] = df['Timestamp'].dt.month
    df['year'] = df['Timestamp'].dt.year
    df['date'] =  df['Timestamp'].dt.date

    # Map numeric months to month names
    month_dict = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 
                7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
    df['month'] = df['month'].map(month_dict)

    st.title('Line Graph')
    fig = px.line(df, x=df.index, y='Residual Demand')
    st.plotly_chart(fig)

    fig2 = plt.figure(figsize=(30,10))

    # Set the plot title and axis labels
    plt.title('Planned and unplanned power cuts BY month')
    plt.xlabel('Year')
    plt.ylabel('Total Power cuts')
    plt.xticks(rotation=45)

    # Show the plot
    st.pyplot(fig2)

    # Reshape the data into a long format
    df = pd.melt(df, id_vars=['year'], value_vars=['Thermal Generation','Dispatchable Generation',
       'Nuclear Generation', 'Eskom Gas Generation', 'Eskom OCGT Generation',
       'Hydro Water Generation', 'Pumped Water Generation'], 
                  var_name='category', value_name='ann')

    # Set the plot title and axis labels
    st.title('Total International exports per year')
    plt.title('Imports and Exports')
    plt.xlabel('Year')
    plt.ylabel('Inports/Exports (Gwh)')


    image_path = "/Users/da_learner_m1_19/Desktop/Screenshot 2023-04-24 at 11.25.02.png"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image)
    else:
        st.write("Error: The specified image file does not exist.")

    st.title('Total Schedulled Power cuts Versus Unschedulled Power Cuts Occured ')
    image_path = "/Users/da_learner_m1_19/Desktop/Screenshot 2023-04-24 at 10.41.42.png"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image)
    else:
        st.write("Error: The specified image file does not exist.")  

    st.title('Nuclear Generation From 2018 to 2023')
    image_path = "/Users/da_learner_m1_19/Desktop/Screenshot 2023-04-24 at 11.37.50.png"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image)
    else:
        st.write("Error: The specified image file does not exist.")

    st.title('Nuclear Generation From 2018 to 2023')
    image_path = "/Users/da_learner_m1_19/Desktop/Screenshot 2023-04-24 at 10.41.42.png"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image)
    else:
        st.write("Error: The specified image file does not exist.")            

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
