import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

print("Loading dataset...")
raw_df = pd.read_csv("SAML-D.csv")

# Sample data to make training faster for UI deployment (while preserving all laundering cases)
print("Sampling data to speed up the training process...")
laundering = raw_df[raw_df['Is_laundering'] == 1]
normal = raw_df[raw_df['Is_laundering'] == 0].sample(n=100000, random_state=42)
df = pd.concat([laundering, normal]).sample(frac=1, random_state=42) # shuffle
0
print(f"Final training dataset size: {len(df)} rows")

# Preprocessing steps matching the original notebook
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

df['Amount'] = np.log1p(df['Amount'])

cols_to_drop = ['Time', 'Date', 'Laundering_type']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

X = df.drop(columns=["Is_laundering"])
y = df["Is_laundering"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

transformer = ColumnTransformer(transformers=[
    ("OrdinalEncoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
    ("RobustScaler", RobustScaler(), numerical_cols)
], remainder="passthrough")

print("Fitting the ColumnTransformer...")
X_transformed = transformer.fit_transform(X)

print("Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_transformed, y)

print("Training the XGBClassifier...")
xgb = XGBClassifier(eval_metric='logloss', max_depth=8, learning_rate=0.2, random_state=42)
xgb.fit(X_res, y_res)

print("Exporting model and transformer to disk...")
joblib.dump(xgb, 'model.pkl')
joblib.dump(transformer, 'transformer.pkl')
print("Successfully generated 'model.pkl' and 'transformer.pkl'. You can now run `streamlit run app.py`.")
