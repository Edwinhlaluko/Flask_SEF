from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the models
model1 = tf.keras.models.load_model('Capacity_model_24.h5')
model2 = tf.keras.models.load_model('Demand_model_24.h5')

data = pd.read_csv('ESK2033.csv')

# Load the scalers
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler1.fit_transform(data)
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler2.fit_transform(data)

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for model 1 predictions
@app.route('/model1', methods=['POST'])
def predict_model1():
    # Get the data from the form
    data = request.form['data']

    # Preprocess the data
    data = pd.DataFrame(data)
    data = scaler1.transform(data)
    X = np.array(data)

    # Make a prediction
    prediction = model1.predict(X)

    # Return the result as JSON
    return jsonify({'result': prediction.tolist()})

# Define the route for model 2 predictions
@app.route('/model2', methods=['POST'])
def predict_model2():
    # Get the data from the form
    data = request.form['data']

    # Preprocess the data
    data = pd.DataFrame(data)
    data = scaler2.transform(data)
    X = np.array(data)

    # Make a prediction
    prediction = model2.predict(X)

    # Return the result as JSON
    return jsonify({'result': prediction.tolist()})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
