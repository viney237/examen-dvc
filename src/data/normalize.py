from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')

# DROP the date column — it's not a useful feature for prediction
X_train = X_train.drop(columns=["date"], errors="ignore")
X_test = X_test.drop(columns=["date"], errors="ignore")

# Create scaler
scaler = StandardScaler()

# Fit on training data ONLY, then transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Note: transform, NOT fit_transform!

# Save scaled data
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('data/processed_data/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed_data/X_test_scaled.csv', index=False)

# Save the scaler for future use (e.g., in production)
joblib.dump(scaler, 'data/processed_data/scaler.pkl')