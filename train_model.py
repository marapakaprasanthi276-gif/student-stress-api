import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("student_stress_dataset_1000_rows.csv")

# Features
X = df[
    [
        "Study_Hours",
        "Sleep_Hours",
        "Attendance_Percentage",
        "CGPA",
        "Social_Media_Hours",
        "Physical_Activity_Hours",
        "Family_Pressure",
        "Financial_Stress",
        "Exam_Score"
    ]
]

# Target
y = df["Stress_Level"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "stress_model.pkl")

print("Model trained successfully!")