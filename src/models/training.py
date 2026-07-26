import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()

# Load best params
best_params = joblib.load('models/best_params.pkl')

# Create model with best params
final_model = RandomForestRegressor(**best_params, random_state=42)

# Train on ALL training data
final_model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(final_model, 'models/best_model.pkl')