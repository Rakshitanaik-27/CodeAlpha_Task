import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset (ensure 'iris.csv' is in this folder)
df = pd.read_csv('iris.csv')

# --- Data Cleaning section ---
df = df.drop_duplicates()
df = df.dropna()
df = df.drop('Id', axis=1)  # Drop ID column

# Encode target label
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])  # 0, 1, 2

X = df.drop('Species', axis=1)
y = df['Species']

# --- Data Training section ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# Save model and label encoder
joblib.dump(model, "iris_model.pkl")
joblib.dump(le, "iris_label_encoder.pkl")
print("Model & label encoder saved.")
