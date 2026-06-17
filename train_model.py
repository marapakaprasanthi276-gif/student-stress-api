import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Example dataset (you can replace with your real dataset)
data = {
    "study_hours": [1,2,3,4,5,6,7,8],
    "sleep_hours": [8,7,6,5,4,3,2,1],
    "screen_time": [2,3,4,5,6,7,8,9],
    "stress_level": [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["study_hours", "sleep_hours", "screen_time"]]
y = df["stress_level"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "stress_model.pkl")

print("Model created successfully!")