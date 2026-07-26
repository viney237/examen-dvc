from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import pandas as pd

X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()  # Flatten to 1D array for sklearn

# Define the model
model = RandomForestRegressor(random_state=42)

# Define the "menu" of settings to try
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# GridSearch with cross-validation (5-fold)
grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

# Save the best parameters
joblib.dump(grid.best_params_, 'models/best_params.pkl')
print("Best params:", grid.best_params_)