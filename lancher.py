from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import radians, sin, cos, sqrt, atan2


app = Flask(__name__)

# Function to calculate the distance between two geographic points
def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Average radius of the Earth in meters
    radius = 6371.0 * 1000

    # Calculate the differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula for distance calculation
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in meters
    distance = radius * c
    return distance

# Load the dataset
df = pd.read_csv("NYC_taxi_trip.csv")

# Select relevant columns
selected_columns = ['vendor_id', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'trip_duration']

# Calculate the trip distances
df['distance'] = df.apply(lambda row: calculate_distance(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

# Split the data into training and testing sets
X = df[['distance', 'vendor_id']]
y = df['trip_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression() # Train the linear regression model
model.fit(X_train, y_train)
@app.route('/explore', methods=['POST'])
def explore():
    data = request.get_json()
    user_latitude = float(data['user_latitude'])
    user_longitude = float(data['user_longitude'])
    dest_latitude = float(data['destination_latitude'])
    dest_longitude = float(data['destination_longitude'])

    # Calculate the distance between user and destination coordinates
    distance = calculate_distance(user_latitude, user_longitude, dest_latitude, dest_longitude)

    # Normalize the user distance between 0 and 1
    normalized_distance = (distance - df['distance'].min()) / (df['distance'].max() - df['distance'].min())

    # Predict the trip duration for vendor 1
    input_data_1 = pd.DataFrame({'distance': [distance], 'vendor_id': [1]})
    predicted_duration_1 = model.predict(input_data_1)

    # Predict the trip duration for vendor 2
    input_data_2 = pd.DataFrame({'distance': [distance], 'vendor_id': [2]})
    predicted_duration_2 = model.predict(input_data_2)

    # Compare the predicted trip durations
    results = {}
    if predicted_duration_1 < predicted_duration_2:
        results['recommendation'] = 1
        results['predicted_duration'] = float(predicted_duration_1)
    elif predicted_duration_1 > predicted_duration_2:
        results['recommendation'] = 2
        results['predicted_duration'] = float(predicted_duration_2)
    else:
        results['recommendation'] = "Both vendors are predicted to have a similar trip duration."
        results['predicted_duration'] = float(predicted_duration_1)

    return jsonify(results)

# Home page
@app.route('/')
def index():
    return render_template('desktop-1.html')

if __name__ == '__main__':
    app.run(debug=True)
