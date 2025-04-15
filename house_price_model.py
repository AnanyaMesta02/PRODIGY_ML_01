import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('train.csv')

# Display the first few rows and information about the data
print("First few rows of the data:")
print(data.head())
print("\nInformation about the data:")
print(data.info())
print("\nSummary Statistics of Numerical Features:")
print(data.describe())

# Visualize the distribution of SalePrice
sns.histplot(data['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.show()

# Check skewness and kurtosis
print("\nSkewness of SalePrice:", data['SalePrice'].skew())
print("Kurtosis of SalePrice:", data['SalePrice'].kurt())

# Select relevant features for the model
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']  # Adjust as needed
X = data[features]
y = data['SalePrice']

# Handle missing values (if any)
X = X.fillna(X.mean())  # Simple imputation; adjust as necessary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R-squared:", r2)

# Optional: Visualize the predictions vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted SalePrice')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
plt.show()