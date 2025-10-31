#LightGBM for DAM-DCL price prediction
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('DAM_DCL_Merged_Prices.csv')

features = ['Average DAM price in EFA block']
target = 'DCL_Clearing_Price'

X = df[features]
y = df[target]

split_index = int(0.8 * len(df)) #80:20 test train split
split_date = df.iloc[split_index]['Delivery day']

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"\nData Split:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing: {X_test.shape[0]} samples")

lgb_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbose=-1
)

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5) #5-folds cross validation
cv_scores = cross_val_score(
    lgb_model, X_train, y_train, 
    cv=tscv, scoring='r2', n_jobs=-1
)


lgb_model.fit(
    X_train, y_train
)

y_pred = lgb_model.predict(X_test)

# key parameters from the prediction model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Plotting results 
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual DCL Clearing Price')
plt.ylabel('Predicted DCL Clearing Price')
plt.title(f'Actual vs Predicted (R-squared = {r2:.3f})')
plt.grid(True, alpha=0.3)
plt.show()

# Model Summary
print(f"LightGBM summary")
print(f"{'='*50}")
print(f"Model: LightGBM Regressor")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Cross-validation R-squared: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Test R-squared: {r2:.4f} ({r2*100:.1f}% variance explained)")
print(f"Test MAE: ±{mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

