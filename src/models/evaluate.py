import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import joblib

X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv').values.ravel()
X_test = pd.read_csv('data/processed_data/X_test.csv')

# Load trained model
model = joblib.load('models/best_model.pkl')

# Make predictions on test set
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Save predictions
results = X_test.copy()
results['actual_silica'] = y_test
results['predicted_silica'] = y_pred
results['error'] = results['actual_silica'] - results['predicted_silica']
results['absolute_error'] = results['error'].abs()
results.to_csv('data/processed_data/predictions.csv', index=False)

# Save metrics
scores = {
    'MSE': mse,
    'RMSE': rmse,
    'R2': r2,
    'MAE': mae
}
with open('metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

print(f"R² Score: {r2:.4f}")