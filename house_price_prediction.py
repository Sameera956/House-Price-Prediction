import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("House_Price.csv")

# Features and target
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
price = model.predict([[2000, 3, 2]])

print("Predicted Price:", price[0])
