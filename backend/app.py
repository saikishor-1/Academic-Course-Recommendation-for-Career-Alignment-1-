from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


app = Flask(__name__)
CORS(app)


df = pd.read_csv("academic_course_recommendation_dataset.csv")


np.random.seed(42)
noise_fraction = 0.15
noise_indices = np.random.choice(df.index, int(len(df)*noise_fraction), replace=False)
course_labels = df["Recommended_Course"].unique()
for idx in noise_indices:
    df.at[idx, "Recommended_Course"] = np.random.choice(course_labels)


label_encoders = {}
for col in ["Interest_Domain", "Career_Goal", "Recommended_Course"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


df["Skill_Score"] = df["Python"] + df["Machine_Learning"] + df["SQL"]
df["Academic_Strength"] = df["CGPA"] * df["Projects"]

X = df.drop("Recommended_Course", axis=1)
y = df["Recommended_Course"]


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X, y)




y_pred = model.predict(X)
train_accuracy = accuracy_score(y, y_pred)


cv_scores = cross_val_score(model, X, y, cv=5)
cv_accuracy = cv_scores.mean()

print("\n===============================")
print(f"Training Accuracy: {round(train_accuracy * 100, 2)} %")
print(f"Cross-Validation Accuracy: {round(cv_scores.mean() * 100, 2)} %")
print("===============================\n")

@app.route("/")
def home():
    return "Academic Course Recommendation API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df_new = pd.DataFrame([data])

    df_new["Interest_Domain"] = label_encoders["Interest_Domain"].transform(
        df_new["Interest_Domain"]
    )
    df_new["Career_Goal"] = label_encoders["Career_Goal"].transform(
        df_new["Career_Goal"]
    )

    df_new["Skill_Score"] = df_new["Python"] + df_new["Machine_Learning"] + df_new["SQL"]
    df_new["Academic_Strength"] = df_new["CGPA"] * df_new["Projects"]

    probs = model.predict_proba(df_new)[0]
    top_indices = np.argsort(probs)[-3:][::-1]

    results = []
    for idx in top_indices:
        course = label_encoders["Recommended_Course"].inverse_transform([idx])[0]
        results.append({
            "course": course,
            "confidence": round(probs[idx]*100, 2)
        })
    print("Received Input:", data)
    print("Prediction Output:", results)
    

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
