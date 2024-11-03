# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the Dataset
file_path = r"C:\Users\user\Downloads\Titanic-Dataset (1).csv"
titanic_df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(titanic_df.head())

# Step 3: Data Preprocessing
# Drop irrelevant columns
titanic_df = titanic_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
titanic_df['Age'] = imputer.fit_transform(titanic_df[['Age']])
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])

# Encode categorical variables
label_encoder = LabelEncoder()
titanic_df['Sex'] = label_encoder.fit_transform(titanic_df['Sex'])
titanic_df['Embarked'] = label_encoder.fit_transform(titanic_df['Embarked'])

# Step 4: Feature Selection
X = titanic_df.drop(columns='Survived')  # Features
y = titanic_df['Survived']  # Target variable

# Step 5: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the model: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 8: Save the Predictions to a CSV file
output_df = pd.DataFrame({'PassengerId': X_test.index, 'Survived_Prediction': y_pred})
output_path = r"C:\Users\user\Downloads\Titanic_Predictions.csv"
output_df.to_csv(output_path, index=False)

print(f"\nPredictions have been saved to {output_path}")
