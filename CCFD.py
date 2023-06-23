# Import the required libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv('credit_card_data.csv')

# Split the data into features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict using the Random Forest classifier
rf_predictions = rf_classifier.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:")
print(rf_confusion_matrix)

# CART algorithm
cart_classifier = DecisionTreeClassifier(random_state=42)
cart_classifier.fit(X_train, y_train)

# Predict using the CART classifier
cart_predictions = cart_classifier.predict(X_test)

# Evaluate the CART model
cart_accuracy = accuracy_score(y_test, cart_predictions)
cart_confusion_matrix = confusion_matrix(y_test, cart_predictions)

print("CART Accuracy:", cart_accuracy)
print("CART Confusion Matrix:")
print(cart_confusion_matrix)