import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
df = pd.read_csv("PS2_Dataset.csv")
df.columns = df.columns.str.strip()

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df.drop(columns=['Suggested Job Role'])
y = df['Suggested Job Role']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert target to categorical (one-hot)
y_encoded = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, stratify=y, test_size=0.2, random_state=42)

# Build MLP model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Deep Learning Model Accuracy: {accuracy:.2f}")

# Save model and encoders
model.save("career_dl_model.h5")
joblib.dump(label_encoders, "label_encoders_dl.pkl")
joblib.dump(scaler, "scaler_dl.pkl")
