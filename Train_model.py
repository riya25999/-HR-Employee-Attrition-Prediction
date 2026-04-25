import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("HR-Employee-Attrition.csv")

# Convert target
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

# Drop useless columns (important)
df = df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis=1)

# Convert categorical to numbers
df = pd.get_dummies(df)

# Split
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model properly
with open("lr_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ New lr_model.pkl created")