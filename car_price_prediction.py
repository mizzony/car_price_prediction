import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from category_encoders import TargetEncoder
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Load dataset
url = "https://raw.githubusercontent.com/mizzony/car_price_prediction/refs/heads/main/used_cars.csv"
data = pd.read_csv(url)

# Clean price and milage
data['price'] = data['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
data['price'] = pd.to_numeric(data['price'], errors='coerce').astype('float64')
data['milage'] = data['milage'].str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False).str.strip()
data['milage'] = pd.to_numeric(data['milage'], errors='coerce').astype('float64')

# Cap outliers at percentiles
price_lower, price_upper = data['price'].quantile([0.01, 0.95])
milage_lower, milage_upper = data['milage'].quantile([0.01, 0.99])
data['price'] = data['price'].clip(lower=price_lower, upper=price_upper)
data['milage'] = data['milage'].clip(lower=milage_lower, upper=milage_upper)

# Handle outliers
data = data[(data['price'] >= 1000) & (data['price'] <= 100000) & 
            (data['milage'] >= 0) & (data['milage'] <= 300000)]
data = data.dropna(subset=['price', 'milage'])

# Fix typo in accident
data['accident'] = data['accident'].replace({'Non': 'None reported'}, regex=True)

# Feature Engineering
data['age'] = 2025 - data['model_year']
data['milage_per_year'] = data['milage'] / (data['age'] + 1)
luxury_brands = ['Porsche', 'BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'INFINITI']
data['luxury_brand'] = data['brand'].isin(luxury_brands).astype(int)
if 'engine' in data.columns:
    data['horsepower'] = data['engine'].str.extract(r'(\d+\.\d+HP)')[0].str.replace('HP', '').astype(float, errors='ignore')
    data['engine_liters'] = data['engine'].str.extract(r'(\d+\.\d+L)')[0].str.replace('L', '').astype(float, errors='ignore')
    data['is_turbo'] = data['engine'].str.contains('Turbo', case=False, na=False).astype(int)
    data['horsepower'] = data['horsepower'].fillna(data['horsepower'].median())
    data['engine_liters'] = data['engine_liters'].fillna(data['engine_liters'].median())
else:
    data['horsepower'] = np.nan
    data['engine_liters'] = np.nan
    data['is_turbo'] = 0
ultra_premium_brands = ['Porsche', 'Mercedes-Benz', 'BMW', 'Audi', 'Lexus']
ultra_premium_models = data[(data['price'] > data['price'].quantile(0.95)) | (data['brand'].isin(ultra_premium_brands))]['model'].unique()
data['ultra_premium'] = (data['model'].isin(ultra_premium_models) | data['brand'].isin(ultra_premium_brands)).astype(int)
data['luxury_horsepower'] = data['luxury_brand'] * data['horsepower']
data['low_milage_premium'] = (data['milage'] < 20000).astype(int) * data['ultra_premium']
model_price_groups = data.groupby('model')['price'].mean().apply(lambda x: 'Low' if x < 20000 else 'Mid' if x < 40000 else 'High')
data['model_price_group'] = data['model'].map(model_price_groups)

# Handle rare models
model_counts = data['model'].value_counts()
rare_model_threshold = len(data) * 0.01
rare_models = model_counts[model_counts < rare_model_threshold].index
data['model'] = data['model'].replace(rare_models, 'Other')

# Define columns
categorical_cols = ['brand', 'model', 'fuel_type', 'transmission', 'accident', 'clean_title', 'model_price_group']
numerical_cols = ['milage', 'age', 'milage_per_year', 'horsepower', 'engine_liters', 'luxury_brand', 'is_turbo', 'ultra_premium', 'luxury_horsepower', 'low_milage_premium']
categorical_cols_for_onehot = [col for col in categorical_cols if col != 'model']
numerical_cols_for_imputation = numerical_cols

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols_for_imputation),
        ('cat_target_encode_model', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('target_enc', TargetEncoder(smoothing=20.0))
        ]), ['model']),
        ('cat_onehot', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols_for_onehot)
    ],
    remainder='passthrough'
)

# Prepare data
X = data.drop(['price', 'engine', 'ext_col', 'int_col'], axis=1, errors='ignore')
y = np.log1p(data['price'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_model.fit(X_train, y_train)
rf_y_pred = np.expm1(rf_model.predict(X_test))
rf_y_test_orig = np.expm1(y_test)
rf_mae = mean_absolute_error(rf_y_test_orig, rf_y_pred)
rf_r2 = r2_score(rf_y_test_orig, rf_y_pred)
print("Random Forest MAE:", rf_mae)
print("Random Forest R2:", rf_r2)

# Fit XGBoost with advanced tuning
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])
param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__learning_rate': [0.01, 0.05],
    'regressor__max_depth': [4, 6],
    'regressor__subsample': [0.7, 0.9],
    'regressor__colsample_bytree': [0.7, 0.9]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_xgb_model = grid_search.best_estimator_
xgb_y_pred = np.expm1(best_xgb_model.predict(X_test))
xgb_mae = mean_absolute_error(rf_y_test_orig, xgb_y_pred)
xgb_r2 = r2_score(rf_y_test_orig, xgb_y_pred)
print("Advanced Tuned XGBoost MAE:", xgb_mae)
print("Advanced Tuned XGBoost R2:", xgb_r2)
print("Best Parameters:", grid_search.best_params_)

# Fit LightGBM with tuning
lgbm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(random_state=42))
])
param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [6, -1],
    'regressor__num_leaves': [31, 50]
}
grid_search = GridSearchCV(lgbm_model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_lgbm_model = grid_search.best_estimator_
lgbm_y_pred = np.expm1(best_lgbm_model.predict(X_test))
lgbm_mae = mean_absolute_error(rf_y_test_orig, lgbm_y_pred)
lgbm_r2 = r2_score(rf_y_test_orig, lgbm_y_pred)
print("Tuned LightGBM MAE:", lgbm_mae)
print("Tuned LightGBM R2:", lgbm_r2)
print("Best Parameters:", grid_search.best_params_)

# Ensemble with adjustment
ensemble_pred = (xgb_y_pred + lgbm_y_pred) / 2
ensemble_pred = np.where(X_test['ultra_premium'] == 1, ensemble_pred * 1.15, ensemble_pred)
ensemble_mae = mean_absolute_error(rf_y_test_orig, ensemble_pred)
ensemble_r2 = r2_score(rf_y_test_orig, ensemble_pred)
print("Adjusted Ensemble MAE:", ensemble_mae)
print("Adjusted Ensemble R2:", ensemble_r2)

# Corrected CV MAE
cv_scores = cross_val_score(best_xgb_model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_mae_log = -cv_scores.mean()
cv_mae_dollars = mean_absolute_error(np.expm1(y_test), np.expm1(y_test - cv_mae_log))
print("Corrected CV MAE (dollars, approx.):", cv_mae_dollars)

# Feature importance
importances = best_xgb_model.named_steps['regressor'].feature_importances_
feature_names = numerical_cols + ['model_target_encoded'] + list(best_xgb_model.named_steps['preprocessor'].transformers_[2][1].named_steps['onehot'].get_feature_names_out(categorical_cols_for_onehot))
importance_dict = dict(zip(feature_names, importances))
top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 Features:", top_features)

# Analyze high-error predictions
errors = np.abs(rf_y_test_orig - ensemble_pred)
high_error_idx = errors > 10000
high_error_data = pd.DataFrame({
    'Actual': rf_y_test_orig[high_error_idx],
    'Predicted': ensemble_pred[high_error_idx],
    'Error': errors[high_error_idx]
}, index=X_test.index[high_error_idx])
print("High-error samples (Error > $10,000):\n", high_error_data)
if not high_error_data.empty:
    print("High-error sample features:\n", X_test.loc[high_error_idx])

# Save best model
mae_scores = {'Random Forest': rf_mae, 'XGBoost': xgb_mae, 'LightGBM': lgbm_mae, 'Ensemble': ensemble_mae}
best_model_name = min(mae_scores, key=mae_scores.get)
best_model = {'Random Forest': rf_model, 'XGBoost': best_xgb_model, 'LightGBM': best_lgbm_model, 'Ensemble': best_xgb_model}[best_model_name]
joblib.dump(best_model, 'best_car_price_model.pkl', compress=9)
print(f"Best model saved: {best_model_name} with MAE {mae_scores[best_model_name]:.2f}")

# Error distribution plot
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=50)
plt.xlabel('Absolute Error ($)')
plt.ylabel('Count')
plt.title('Error Distribution (Ensemble)')
plt.grid(True)
plt.savefig('error_distribution.png')