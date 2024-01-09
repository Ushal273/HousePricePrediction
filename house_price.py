import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load data
housing = pd.read_csv("D:\HousePricePrediction\kathmandudata.csv")

# Dropping unnecessary columns
housing.drop(['location', 'dhur', 'Aana', 'Paisa', 'Daam', 'Years ago'], axis=1, inplace=True)

# Replacing 'N/A' values with NaN
housing = housing.replace('N/A', np.nan)

# Calculating average values for missing 'floors' and 'bedroom' values
avg_floors = housing['floors'].mean().round(0)
avg_bathroom = housing['bedroom'].mean().round(0)

# Replacing null values with average values
housing['floors'] = housing['floors'].replace(np.nan, avg_floors)
housing['bathroom'] = housing['bedroom'].replace(0, 1)

# Calculating Interquartile Range (IQR) for 'Price'
q3, q1 = np.percentile(housing['Price'], [75, 25])
price_iqr = q3 - q1

# Encode 'Road type' using LabelEncoder
label_encoder = LabelEncoder()
housing['Road type'] = label_encoder.fit_transform(housing['Road type'])

# Splitting the data into training and testing sets
X = housing.drop("Price", axis=1)
y = housing["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training a Random Forest Regressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Making predictions on the test set
predictions = model.predict(X_test)

# Calculate RMSE and RMSE as a percentage of the Price range
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

min_price = y.min()
max_price = y.max()
price_range = max_price - min_price

accuracy_percentage = (1 - rmse / price_range) * 100

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Accuracy Percentage:", accuracy_percentage)

# Visualizing actual vs. predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([min_price, max_price], [min_price, max_price], 'r-', lw=2)  # Line of perfect predictions
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.grid(True)
plt.show()

''''
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class RandomForestRegressorCustom:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_subset, y_subset = X.iloc[indices], y.iloc[indices]
            tree = self.build_tree(X_subset, y_subset, depth=0)
            self.trees.append(tree)

    def build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            return {'prediction': np.mean(y)}

        feature_idx, threshold, split_gain = self.find_best_split(X, y)
        left_idxs = (X.iloc[:, feature_idx] <= threshold).values
        right_idxs = (X.iloc[:, feature_idx] > threshold).values

        left = self.build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self.build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {'feature_idx': feature_idx, 'threshold': threshold, 'split_gain': split_gain,
                'left': left, 'right': right}

    def find_best_split(self, X, y):
        best_feature, best_threshold, best_gain = None, None, -float('inf')
        for feature in X.columns:
            values = np.unique(X[feature])
            for value in values:
                if isinstance(value, (int, float)):
                    left_idxs = X[feature] <= value
                    right_idxs = X[feature] > value
                else:
                    left_idxs = X[feature] == value
                    right_idxs = ~left_idxs

                gain = self.gini_impurity(y) - (len(y[left_idxs]) / len(y) * self.gini_impurity(y[left_idxs]) +
                                                len(y[right_idxs]) / len(y) * self.gini_impurity(y[right_idxs]))
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, value, gain
        return best_feature, best_threshold, best_gain

    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
        classes = Counter(y)
        impurity = 1
        for c in classes:
            prob = classes[c] / len(y)
            impurity -= prob ** 2
        return impurity

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(self.predict_tree(X.copy(), tree))
        return np.mean(predictions, axis=0)

def predict_tree(self, X, tree):
    if 'prediction' in tree:
        return np.array([tree['prediction']] * len(X))

    feature_idx, threshold = tree['feature_idx'], tree['threshold']
    left_idxs = X.iloc[:, feature_idx] <= threshold
    right_idxs = X.iloc[:, feature_idx] > threshold

    left_indices = np.where(left_idxs)[0]
    right_indices = np.where(right_idxs)[0]

    left_predictions = self.predict_tree(X.iloc[left_indices].copy(), tree['left'])
    right_predictions = self.predict_tree(X.iloc[right_indices].copy(), tree['right'])

    combined_predictions = np.zeros(len(X))
    combined_predictions[left_indices] = left_predictions
    combined_predictions[right_indices] = right_predictions
    return combined_predictions

housing = pd.read_csv("D:\HousePricePrediction\kathmandudata.csv")

# Dropping unnecessary columns
columns_to_drop = ['location', 'dhur', 'Aana', 'Paisa', 'Daam', 'Years ago']
housing = housing.drop(columns_to_drop, axis=1, errors='ignore')

# Replacing 'N/A' values with NaN
housing = housing.replace('N/A', np.nan)

# Calculating average values for missing 'floors' and 'bedroom' values
avg_floors = housing['floors'].mean().round(0)
avg_bathroom = housing['bedroom'].mean().round(0)

# Replacing null values with average values
housing['floors'] = housing['floors'].replace(np.nan, avg_floors)
housing['bathroom'] = housing['bedroom'].replace(0, 1)

# Encode 'Road type' using LabelEncoder
labelencoder = LabelEncoder()
housing['Road type'] = labelencoder.fit_transform(housing['Road type'])

# Handle 'N/A' values in 'bathroom'
housing['bathroom'] = housing['bathroom'].replace('N/A', avg_bathroom)

# Encode categorical columns
if 'location' in housing.columns:
    housing['location'] = labelencoder.fit_transform(housing['location'])

# Splitting data into features (X) and target variable (y)
X = housing.drop('Price', axis=1)
y = housing['Price']

# Initialize and fit the RandomForestRegressorCustom
model = RandomForestRegressorCustom(n_estimators=100, max_depth=None)
model.fit(X, y)

# Example new data for prediction
new_data = pd.DataFrame({
    'bedroom': [0],
    'bathroom': [1],
    'floors': [3],
    'parking': [0],
    'roadsize(feet)': [20],
    'Road type': [5],
    'Area(sq ft)': [2566.87]
})

# Make predictions
test_prediction = model.predict(new_data)
print(test_prediction)
'''