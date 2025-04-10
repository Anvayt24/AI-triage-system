import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("heart_disease_uci.csv")

symptoms = ['cough', 'injury', 'diarrhea', 'high fever', 'rash', 'vomiting', 
            'shortness of breath', 'severe weakness', 'sweating excessively']
probs = [0.15, 0.12, 0.15, 0.12, 0.10, 0.10, 0.10, 0.08, 0.08]
df['symptoms'] = np.random.choice(symptoms, size=len(df), p=probs)


df['temp'] = np.random.uniform(36, 37, size=len(df))  # Normal baseline
df['resp_rate'] = np.random.randint(12, 20, size=len(df))  # Normal breathing
for i, row in df.iterrows():
    if row['symptoms'] == 'high fever':
        df.at[i, 'temp'] = np.random.uniform(38, 42)  # Heat stroke, infections
        df.at[i, 'trestbps'] = np.random.randint(100, 150)
        df.at[i, 'thalch'] = np.random.randint(100, 140)
    elif row['symptoms'] == 'diarrhea':
        df.at[i, 'temp'] = np.random.uniform(37.5, 39)  # Mild fever, dehydration
        df.at[i, 'trestbps'] = np.random.randint(80, 110)  # Lower BP
        df.at[i, 'thalch'] = np.random.randint(90, 120)
    elif row['symptoms'] == 'vomiting':
        df.at[i, 'temp'] = np.random.uniform(37, 38.5)  # Food poisoning
        df.at[i, 'trestbps'] = np.random.randint(90, 120)
        df.at[i, 'thalch'] = np.random.randint(100, 130)
    elif row['symptoms'] == 'shortness of breath':
        df.at[i, 'resp_rate'] = np.random.randint(20, 35)  # Asthma, heat stress
        df.at[i, 'trestbps'] = np.random.randint(110, 160)
        df.at[i, 'thalch'] = np.random.randint(100, 140)
    elif row['symptoms'] == 'severe weakness':
        df.at[i, 'temp'] = np.random.uniform(36, 38)  # Dehydration
        df.at[i, 'trestbps'] = np.random.randint(70, 100)  # Low BP
        df.at[i, 'thalch'] = np.random.randint(90, 120)
    elif row['symptoms'] == 'sweating excessively':
        df.at[i, 'temp'] = np.random.uniform(37, 40)  # Heat exhaustion
        df.at[i, 'trestbps'] = np.random.randint(90, 130)
        df.at[i, 'thalch'] = np.random.randint(100, 140)
    elif row['symptoms'] == 'rash':
        df.at[i, 'temp'] = np.random.uniform(36.5, 38)  # Heat rash, dengue
        df.at[i, 'trestbps'] = np.random.randint(100, 140)
        df.at[i, 'thalch'] = np.random.randint(90, 120)
    elif row['symptoms'] == 'cough':
        df.at[i, 'resp_rate'] = np.random.randint(18, 25)  # Mild respiratory
        df.at[i, 'thalch'] = np.random.randint(90, 110)
    elif row['symptoms'] == 'injury':
        df.at[i, 'trestbps'] = np.random.randint(80, 120)
        df.at[i, 'thalch'] = np.random.randint(90, 130)

df['chol'] = df['chol'].apply(lambda x: np.random.randint(150, 250) if x > 200 or x == 0 else x)

def assign_triage(row):
    if (row['trestbps'] < 90) or (row['temp'] > 40) or (row['thalch'] > 140) or \
       (row['resp_rate'] > 25) or (row['symptoms'] in ['high fever', 'shortness of breath'] and row['temp'] > 39) or \
       (row['symptoms'] == 'severe weakness' and row['trestbps'] < 100):
        return 'Red'
    elif (row['temp'] > 38 and row['temp'] <= 40) or (row['trestbps'] > 140 and row['trestbps'] <= 160) or \
         (row['thalch'] > 110 and row['thalch'] <= 140) or (row['resp_rate'] > 20) or \
         (row['symptoms'] in ['diarrhea', 'vomiting', 'sweating excessively', 'shortness of breath']):
        return 'Yellow'
    else:
        return 'Green'

df['triage'] = df.apply(assign_triage, axis=1)
df['age'] = df['age'].apply(lambda x: np.random.randint(20, 50) if x > 50 else x)
df = df.drop(columns=['num', 'ca', 'thal', 'slope', 'restecg', 'exang', 'oldpeak', 'fbs', 'dataset'])

n_target = 2000
n_copies = (n_target // len(df)) + 1
df_scaled = pd.concat([df] * n_copies, ignore_index=True).head(n_target)

# Perturb vitals
df_scaled['trestbps'] += np.random.randint(-5, 6, size=len(df_scaled))
df_scaled['chol'] += np.random.randint(-10, 11, size=len(df_scaled))
df_scaled['thalch'] += np.random.randint(-5, 6, size=len(df_scaled))
df_scaled['temp'] += np.random.uniform(-0.2, 0.2, size=len(df_scaled))
df_scaled['resp_rate'] += np.random.randint(-2, 3, size=len(df_scaled))


df_scaled['triage'] = df_scaled.apply(assign_triage, axis=1)
df_scaled['id'] = range(1, len(df_scaled) + 1)


# Save
df_scaled.to_csv('triage_data_summ.csv', index=False)
print("Tweaked and scaled dataset")
print("Triage distribution:\n", df_scaled['triage'].value_counts())

# Preprocess
features = ['age', 'trestbps', 'chol', 'thalch', 'temp','resp_rate']
categorical_features = ['symptoms']
df_scaled[features] = df_scaled[features].fillna(df_scaled[features].mean())
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df_scaled[features])
X_categorical = pd.get_dummies(df_scaled[categorical_features], prefix='symptoms')  # Fix prefix typo
X = np.hstack((X_numeric, X_categorical))


label_map = {'Green': 0, 'Yellow': 1, 'Red': 2} 
y = df_scaled['triage'].map(label_map)
if y.isnull().any():
    print("Warning: NaN values in y - check triage labels")
y_categorical = to_categorical(y, num_classes=3)

# Split and check distribution
X_train, X_temp, y_train, y_temp = train_test_split(X, y_categorical, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print("Test set distribution:\n", pd.Series(np.argmax(y_test, axis=1)).value_counts())

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), 
callbacks=[early_stopping], verbose=1)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.2f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report
print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes, target_names=['Green', 'Yellow', 'Red']))

# Save
model.save('triage_model_refined2.h5')
import joblib
joblib.dump(scaler, 'scaler_refined2.pkl')