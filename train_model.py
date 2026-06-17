import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("student_stress_dataset_1000_rows.csv")

X = df.drop("Stress_Level", axis=1)
y = df["Stress_Level"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "stress_model.pkl")

print("Model trained successfully!")