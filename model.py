import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

data=pd.read_csv("C:\\Users\\KARUNSAI\\Desktop\\Admission_Predict1.csv")
# Split into features and target
X = data[['GRE _Score','TOEFL _Score','University _Rating','SOP','LOR ','CGPA']]
y = data['Chance of Admit ']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

prediction_input = pd.DataFrame([[337, 118, 4, 4.5, 4.5, 9.65]], columns=['GRE _Score', 'TOEFL _Score', 'University _Rating', 'SOP', 'LOR ', 'CGPA'])

# Make a prediction
prediction = model.predict(prediction_input)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nPredictions vs Actual Values:")
print(results)
# Actual vs Predicted Values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Perfect predictions line
plt.title('Actual vs Predicted: Chance of Admission')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
# Save the trained model using Pickle
with open("chance_of_admission_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
print("Model trained and saved successfully.")