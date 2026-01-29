import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_PATH = "academic_course_recommendation_dataset.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        "Dataset not found. Please place 'academic_course_recommendation_dataset.csv' in the project folder."
    )

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully")
print("Shape:", df.shape)


np.random.seed(42)

noise_fraction = 0.15
noise_indices = np.random.choice(
    df.index,
    size=int(len(df) * noise_fraction),
    replace=False
)

course_labels = df["Recommended_Course"].unique()

for idx in noise_indices:
    df.at[idx, "Recommended_Course"] = np.random.choice(course_labels)

print("Controlled label noise added")


label_encoders = {}
categorical_cols = ["Interest_Domain", "Career_Goal", "Recommended_Course"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


df["Skill_Score"] = df["Python"] + df["Machine_Learning"] + df["SQL"]
df["Academic_Strength"] = df["CGPA"] * df["Projects"]


X = df.drop("Recommended_Course", axis=1)
y = df["Recommended_Course"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Accuracy:", round(cv_scores.mean() * 100, 2), "%")


print("\nFeature Importance:")
importances = model.feature_importances_

for name, score in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {round(score, 3)}")


new_student = {
    "CGPA": 7,
    "Python": 1,
    "Machine_Learning": 0,
    "SQL": 0,
    "Projects": 1,
    "Interest_Domain": "Cybersecurity",
    "Career_Goal": "Industry",
    "Industry_Demand_Score": 8,
    "Internship_Experience": 1
}

new_df = pd.DataFrame([new_student])

new_df["Interest_Domain"] = label_encoders["Interest_Domain"].transform(
    new_df["Interest_Domain"]
)
new_df["Career_Goal"] = label_encoders["Career_Goal"].transform(
    new_df["Career_Goal"]
)

new_df["Skill_Score"] = new_df["Python"] + new_df["Machine_Learning"] + new_df["SQL"]
new_df["Academic_Strength"] = new_df["CGPA"] * new_df["Projects"]


probabilities = model.predict_proba(new_df)[0]
top_indices = np.argsort(probabilities)[-3:][::-1]

print("\nTop 3 Recommended Courses:")
for idx in top_indices:
    course = label_encoders["Recommended_Course"].inverse_transform([idx])[0]
    print(course, "→", round(probabilities[idx] * 100, 2), "%")

print("\n===== PROGRAM EXECUTION COMPLETED SUCCESSFULLY =====")