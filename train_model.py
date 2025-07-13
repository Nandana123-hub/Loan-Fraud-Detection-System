import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ✅ Load your uploaded dataset
df = pd.read_csv("loan_data.csv/train.csv") 

# ✅ Drop Loan_ID as it's just a unique identifier
df.drop("Loan_ID", axis=1, inplace=True)

# ✅ Drop rows with missing values
df = df.dropna()

# ✅ Encode categorical columns
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status', 'Dependents']:
    df[col] = le.fit_transform(df[col])

# ✅ Prepare features and label
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ✅ Scale numerical values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ✅ Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model trained and saved successfully!")
