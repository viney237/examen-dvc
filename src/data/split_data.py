import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/raw_data/raw.csv')

# Separate inputs (X) and target (y)
X = df.drop('silica_concentrate', axis=1)  # Everything except the last column
y = df['silica_concentrate']                # Just the target

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save to data/processed_data/
X_train.to_csv('data/processed_data/X_train.csv', index=False)
X_test.to_csv('data/processed_data/X_test.csv', index=False)
y_train.to_csv('data/processed_data/y_train.csv', index=False)
y_test.to_csv('data/processed_data/y_test.csv', index=False)