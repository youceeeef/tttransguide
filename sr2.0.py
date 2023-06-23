import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import radians, sin, cos, sqrt, atan2

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
data = pd.read_csv("NYC_taxi_trip.csv")

# Select relevant columns
selected_columns = ['vendor_id', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'trip_duration']
data = data[selected_columns]

# Calculate the trip distances
data['distance'] = data.apply(lambda row: calculate_distance(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

# Split the data into training and testing sets
X = data[['distance', 'vendor_id']]
y = data['trip_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the entire dataset
data['predicted_duration'] = model.predict(X)

# Normalize the predicted durations between 0 and 1
normalized_duration = (data['predicted_duration'] - data['predicted_duration'].min()) / (data['predicted_duration'].max() - data['predicted_duration'].min())

# User and destination coordinates
#user_latitude = 40.72300  # User latitude
#user_longitude = -74.0260  # User longitude
#dest_latitude = 40.7168  # Destination latitude
#dest_longitude = -74.0020  # Destination longitude

# Calculate the distance between user and destination coordinates
distance = calculate_distance(user_latitude, user_longitude, dest_latitude, dest_longitude)

# Normalize the user distance between 0 and 1
normalized_distance = (distance - data['distance'].min()) / (data['distance'].max() - data['distance'].min())

# Predict the trip duration for vendor 1
input_data_1 = pd.DataFrame({'distance': [normalized_distance], 'vendor_id': [1]})
predicted_duration_1 = model.predict(input_data_1)

# Predict the trip duration for vendor 2
input_data_2 = pd.DataFrame({'distance': [normalized_distance], 'vendor_id': [2]})
predicted_duration_2 = model.predict(input_data_2)

# Compare the predicted trip durations
if predicted_duration_1 < predicted_duration_2:
    print("Vendor 1 is faster avec le temps :",predicted_duration_1)
elif predicted_duration_1 > predicted_duration_2:
    print("Vendor 2 is faster.",predicted_duration_2)
else:
    print("Both vendors have a similar predicted trip duration.")
