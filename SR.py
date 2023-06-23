import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Charger le dataset
df = pd.read_csv('NYC_taxi_trip.csv')

# Sélectionner les caractéristiques pertinentes pour la recommandation
features = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 'trip_duration']

# Filtrer les colonnes pertinentes
df_filtered = df[features]

# Prendre un échantillon aléatoire de la dataframe
sample_size = 10000  # Specify the desired sample size
df_filtered = df_filtered.sample(n=sample_size, random_state=42)  # Set random_state for reproducibility

# Fonction de calcul de la distance haversine entre deux points géographiques
def calculate_haversine_distance(row):
    pickup_coords = (radians(row['pickup_latitude']), radians(row['pickup_longitude']))
    dropoff_coords = (radians(row['dropoff_latitude']), radians(row['dropoff_longitude']))
    distance = haversine_distances([pickup_coords, dropoff_coords])[0][1] * 6371000  # Radius of Earth in meters
    return distance

# Normaliser les caractéristiques entre 0 et 1
def normalize_features(df):
    normalized_df = df.copy()
    for feature in ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_duration']:
        normalized_df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
    return normalized_df

# Ajouter une colonne pour la distance entre pickup et dropoff
df_filtered['distance'] = df_filtered.apply(calculate_haversine_distance, axis=1)

# Normaliser les caractéristiques
df_normalized = normalize_features(df_filtered)

# Diviser les données en caractéristiques (X) et variable cible (y)
X = df_normalized.drop('trip_duration', axis=1)
y = df_normalized['trip_duration']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle de régression
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Calculer la matrice de similarité basée sur la distance haversine
def calculate_similarity_matrix():
    coordinates = df_normalized[['pickup_latitude', 'pickup_longitude']].apply(
        lambda row: [radians(row['pickup_latitude']), radians(row['pickup_longitude'])], axis=1
    )
    distance_matrix = haversine_distances(coordinates.tolist(), coordinates.tolist()) * 6371000
    similarity_matrix = 1 / (1 + distance_matrix)
    return similarity_matrix

# Entraîner les fournisseurs de taxi en utilisant les trajets dans le dataset
def train_vendors():
    vendor1_trips = df_normalized[df_normalized['vendor_id'] == 1]['trip_duration']
    vendor2_trips = df_normalized[df_normalized['vendor_id'] == 2]['trip_duration']
    vendor1_mean = vendor1_trips.mean()
    vendor2_mean = vendor2_trips.mean()
    return vendor1_mean, vendor2_mean

# Fonction de recommandation hybride
def hybrid_recommendation(user_coordinates, destination_coordinates):
    # Calculer la distance haversine entre l'utilisateur et les points de départ
    user_distance = haversine_distances([user_coordinates, destination_coordinates])[0][1] * 6371000

    # Normaliser la distance entre 0 et 1
    normalized_user_distance = (user_distance - df_normalized['distance'].min()) / (df_normalized['distance'].max() - df_normalized['distance'].min())

    # Calculer les scores de recommandation en utilisant la similarité
    similarity_matrix = calculate_similarity_matrix()

    # Calculer les scores de prédiction de la durée du trajet avec le modèle de régression
    trip_duration_predictions = regression_model.predict(X)
    # Normaliser les scores de prédiction entre 0 et 1
    normalized_trip_duration_predictions = (trip_duration_predictions - min(trip_duration_predictions)) / (max(trip_duration_predictions) - min(trip_duration_predictions))

    # Trier les recommandations par score décroissant
    recommendations = df_normalized.copy()
    recommendations['score'] = normalized_user_distance * 0.3 + normalized_trip_duration_predictions[:len(df_normalized)] * 0.7
    recommendations = recommendations.sort_values('score', ascending=False)

    return recommendations[['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
                            'dropoff_longitude', 'dropoff_latitude', 'trip_duration', 'score']]

# Coordonnées de l'utilisateur
user_latitude = float(input("Enter your latitude: "))
user_longitude = float(input("Enter your longitude: "))
user_coordinates = (radians(user_latitude), radians(user_longitude))

# Coordonnées de la destination
destination_latitude = float(input("Enter the destination latitude: "))
destination_longitude = float(input("Enter the destination longitude: "))
destination_coordinates = (radians(destination_latitude), radians(destination_longitude))

# Obtenir les recommandations pour l'utilisateur
recommended_trips = hybrid_recommendation(user_coordinates, destination_coordinates)

# Entraîner les fournisseurs de taxi
vendor1_mean, vendor2_mean = train_vendors()

# Faire une prédiction sur les coordonnées entrées par l'utilisateur
if vendor1_mean < vendor2_mean:
    recommended_vendor = 1
else:
    recommended_vendor = 2

# Afficher les résultats de recommandation
print("Recommended Vendor: Vendor", recommended_vendor)
print("Recommended Trips:")
print(recommended_trips.head())
