import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated loading of data (replace this line with your actual data loading line)
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes']
}
play_tennis = pd.DataFrame(data)

# Data preprocessing with LabelEncoder
number = LabelEncoder()
play_tennis['Outlook'] = number.fit_transform(play_tennis['Outlook'])
play_tennis['Temperature'] = number.fit_transform(play_tennis['Temperature'])
play_tennis['Humidity'] = number.fit_transform(play_tennis['Humidity'])
play_tennis['Wind'] = number.fit_transform(play_tennis['Wind'])
play_tennis['Play Tennis'] = number.fit_transform(play_tennis['Play Tennis'])

# Features and target variable
features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play Tennis"

# Splitting dataset into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(
    play_tennis[features], play_tennis[target], test_size=0.33, random_state=54
)

# Model initialization and training
model = GaussianNB()
model.fit(features_train, target_train)

# Making predictions and evaluating the model
pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)

# Print outputs
print("Model Accuracy:", accuracy)
print("Predicted Values:", pred)
print("Predicted Class for input [1, 2, 0, 1]:", model.predict([[1, 2, 0, 1]]))
