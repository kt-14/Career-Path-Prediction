import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("PS2_Dataset.csv")
df.columns = df.columns.str.strip()

print(f"Original dataset shape: {df.shape}")

# Better job role mapping to reduce class imbalance
job_role_mapping = {
    'Applications Developer': 'Software Engineer',
    'CRM Technical Developer': 'Software Engineer', 
    'Software Developer': 'Software Engineer',
    'Software Engineer': 'Software Engineer',
    'Database Developer': 'Software Engineer',  
    'Mobile Applications Developer': 'Software Engineer',  
    'Web Developer': 'Software Engineer',  # Group with Software Engineer
    'UX Designer': 'Designer',
    'Software Quality Assurance (QA) / Testing': 'QA Engineer',
    'Network Security Engineer': 'Security Engineer',
    'Systems Security Administrator': 'Security Engineer',  # Group with Security Engineer
    'Technical Support': 'Support Engineer'
}

df['Suggested Job Role'] = df['Suggested Job Role'].map(job_role_mapping)
df = df.dropna(subset=['Suggested Job Role'])

print("Job role distribution after improved mapping:")
print(df['Suggested Job Role'].value_counts())

#  Remove features that are causing overfitting
features_to_remove = [
    'Taken inputs from seniors or elders',
    'Interested Type of Books', 
    'Introvert'
]

# Keep only the most predictive features
important_features = [
    'Logical quotient rating',
    'hackathons', 
    'coding skills rating',
    'public speaking points',
    'self-learning capability?',
    'Extra-courses did',
    'certifications',
    'workshops',
    'interested career area',  # highly predictive
    'Management or Technical',  # highly predictive
    'Suggested Job Role'
]

# Filter to important features only
df_filtered = df[important_features].copy()

print(f"Filtered dataset shape: {df_filtered.shape}")

# Encode categorical columns
label_encoders = {}
for col in df_filtered.select_dtypes(include='object').columns:
    if col != 'Suggested Job Role':
        le = LabelEncoder()
        df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))
        label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df_filtered['Suggested Job Role'] = target_encoder.fit_transform(df_filtered['Suggested Job Role'])
label_encoders['Suggested Job Role'] = target_encoder

print(f"Final classes: {target_encoder.classes_}")
print(f"Final class distribution:")
print(pd.Series(df_filtered['Suggested Job Role']).value_counts())

# Split features and target
X = df_filtered.drop(columns=['Suggested Job Role'])
y = df_filtered['Suggested Job Role']

# Handle class imbalance with SMOTE
print("\n=== Applying SMOTE for class balancing ===")

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42, k_neighbors=3)
X_balanced, y_balanced = smote.fit_resample(X, y)

print(f"Original class distribution: {np.bincount(y)}")
print(f"Balanced class distribution: {np.bincount(y_balanced)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_balanced
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Try multiple algorithms to choose the best

# 1. Random Forest with class balancing
print("\n=== Training Balanced Random Forest ===")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',  
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy:.3f}")

# 2. Gradient Boosting
print("\n=== Training Gradient Boosting ===")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_accuracy = gb_model.score(X_test, y_test)
print(f"Gradient Boosting Accuracy: {gb_accuracy:.3f}")

# 3. Neural Network
print("\n=== Training Improved Neural Network ===")

# Calculate class weights for neural network
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print(f"Class weights: {class_weight_dict}")

# Convert to categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

def create_focused_model(input_dim, num_classes):
    model = Sequential([
        Dense(64, input_shape=(input_dim,), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(), 
        Dropout(0.4),
        
        Dense(16, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Use a smaller learning rate
    optimizer = Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = create_focused_model(X_train.shape[1], y_train_cat.shape[1])

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

# Train with class weights
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate all models
print("\n=== Model Evaluation ===")

nn_loss, nn_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Neural Network Accuracy: {nn_accuracy:.3f}")

# Get predictions
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
nn_pred = np.argmax(model.predict(X_test), axis=1)

# Detailed evaluation
print(f"\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=target_encoder.classes_))

print(f"\nGradient Boosting Classification Report:")
print(classification_report(y_test, gb_pred, target_names=target_encoder.classes_))

print(f"\nNeural Network Classification Report:")
print(classification_report(y_test, nn_pred, target_names=target_encoder.classes_))

# Choose best model
accuracies = {
    'Random Forest': rf_accuracy,
    'Gradient Boosting': gb_accuracy, 
    'Neural Network': nn_accuracy
}

best_model_name = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model_name]

print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.3f}")

# Save the best model
if best_model_name == 'Random Forest':
    joblib.dump(rf_model, "career_best_model.pkl")
    best_model = rf_model
    model_type = 'rf'
elif best_model_name == 'Gradient Boosting':
    joblib.dump(gb_model, "career_best_model.pkl") 
    best_model = gb_model
    model_type = 'gb'
else:
    model.save("career_dl_model.keras")
    best_model = model
    model_type = 'nn'

# Save preprocessing objects
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save model metadata
model_info = {
    'model_type': model_type,
    'accuracy': best_accuracy,
    'feature_columns': list(X.columns),
    'target_classes': list(target_encoder.classes_),
    'used_smote': True
}
joblib.dump(model_info, "model_info.pkl")

print(f"\nFinal Model Accuracy: {best_accuracy:.3f}")
print("Model saved successfully!")

# Feature importance for best tree-based model
if model_type in ['rf', 'gb']:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance ({best_model_name}):")
    print(feature_importance)