import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

os.makedirs("saved_models", exist_ok=True)

# ---------------- DIABETES ----------------
data = pd.read_csv("dataset/diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

print("Diabetes Accuracy:", accuracy_score(y_test, model.predict(X_test)))

pickle.dump(model, open("saved_models/diabetes_model.sav", "wb"))


# ---------------- HEART ----------------
data = pd.read_csv("dataset/heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

print("Heart Accuracy:", accuracy_score(y_test, model.predict(X_test)))

pickle.dump(model, open("saved_models/heart_disease_model.sav", "wb"))


# ---------------- PARKINSONS ----------------
data = pd.read_csv("dataset/parkinsons.csv")

# 🔴 REMOVE STRING COLUMN
data = data.drop("name", axis=1)

X = data.drop("status", axis=1)
y = data["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

print("Parkinsons Accuracy:", accuracy_score(y_test, model.predict(X_test)))

pickle.dump(model, open("saved_models/parkinsons_model.sav", "wb"))

print("\nALL MODELS TRAINED SUCCESSFULLY")
