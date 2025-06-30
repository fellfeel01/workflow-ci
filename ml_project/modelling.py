import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Set MLflow experiment
mlflow.set_experiment("Amazon_Rating_Prediction_Autolog")

# Load the preprocessed dataset
df = pd.read_csv('amazon_preprocessed.csv')

# Drop irrelevant columns
drop_cols = ['product_id', 'product_name', 'about_product', 'user_id', 'user_name', 
             'review_id', 'review_title', 'review_content', 'img_link', 'product_link', 'rating_bin']
df = df.drop(columns=drop_cols, errors='ignore')

# Select features
numerical_cols = ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count']
cat_cols = [col for col in df.columns if col.startswith('cat_')]
features = numerical_cols + cat_cols + ['category_encoded']
X = df[features]
y = df['rating']

# Handle multicollinearity
corr_matrix = X[numerical_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
X = X.drop(columns=to_drop, errors='ignore')
print(f"Dropped highly correlated features: {to_drop}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Start MLflow run
with mlflow.start_run(run_name="RandomForest_Autolog"):
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Print metrics
    print("\nModel Performance:")
    print(f"Training MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"Testing MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")