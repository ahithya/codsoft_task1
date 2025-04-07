# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# @title Default title text
data = pd.read_csv("Titanic-Dataset.csv")
print(data.info())
print(data.head())

# Drop unnecessary columns
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Fill missing Age with median
data["Age"].fillna(data["Age"].median(), inplace=True)

# Fill missing Embarked with most common value
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)

print(data.head())

X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

