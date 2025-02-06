import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# ---------------------------------------------------------------------------------
# Load the dataset
# ---------------------------------------------------------------------------------
DATA_FILE = "data.csv"  # Ensure this file exists in your working directory
try:
    data = pd.read_csv(DATA_FILE)
    print("Dataset loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {str(e)}")
    exit()

# ---------------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------------
FEATURES = [
    'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
    'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'
]
TARGET = 'Response'

# Check for missing columns
for column in FEATURES + [TARGET]:
    if column not in data.columns:
        print(f"Error: Column '{column}' not found in the dataset.")
        exit()

X = data[FEATURES]
y = data[TARGET]

# ---------------------------------------------------------------------------------
# Splitting Data
# ---------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and test sets.")

# ---------------------------------------------------------------------------------
# Manual Preprocessing
# ---------------------------------------------------------------------------------
# Step 1: Handling Categorical Data with OneHotEncoder
categorical_columns = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Apply OneHotEncoder to categorical columns
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

# Convert encoded arrays into DataFrames
encoded_columns = encoder.get_feature_names_out(categorical_columns)
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)

# Step 2: Handling Numerical Data with StandardScaler
numerical_columns = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
scaler = StandardScaler()

# Scale numerical columns
X_train_scaled = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled = scaler.transform(X_test[numerical_columns])

# Convert scaled arrays into DataFrames
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numerical_columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numerical_columns, index=X_test.index)

# Step 3: Combine Processed Features
X_train_final = pd.concat([X_train_encoded_df, X_train_scaled_df, X_train[['Driving_License', 'Previously_Insured']]], axis=1)
X_test_final = pd.concat([X_test_encoded_df, X_test_scaled_df, X_test[['Driving_License', 'Previously_Insured']]], axis=1)

# ---------------------------------------------------------------------------------
# Train the Random Forest Model
# ---------------------------------------------------------------------------------
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_final, y_train)
print("Model training complete!")

# ---------------------------------------------------------------------------------
# Save the Model and Preprocessors
# ---------------------------------------------------------------------------------
MODEL_FILE = "Random Forest_model.joblib"
ENCODER_FILE = "onehot_encoder.joblib"
SCALER_FILE = "scaler.joblib"

print("Saving the trained model and preprocessing tools...")
dump(model, MODEL_FILE)
dump(encoder, ENCODER_FILE)
dump(scaler, SCALER_FILE)

print(f"Model saved as '{MODEL_FILE}'")
print(f"OneHotEncoder saved as '{ENCODER_FILE}'")
print(f"Scaler saved as '{SCALER_FILE}'")
