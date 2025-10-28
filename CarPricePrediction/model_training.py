import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv('car_data.csv')  # Now it's sure this exists!

# ----- DATA CLEANING -----
df = df.drop_duplicates()
df = df.dropna()

# Encode categorical variables
categorical = ['Fuel_Type', 'Selling_type', 'Transmission', 'Car_Name']
for col in categorical:
    if col in df.columns:
        df[col] = df[col].astype('category').cat.codes

# Define features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

joblib.dump(model, 'car_price_model.pkl')
print("Model saved as car_price_model.pkl")
