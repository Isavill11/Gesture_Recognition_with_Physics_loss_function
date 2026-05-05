import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.losses import sparse_categorical_crossentropy

from keras._tf_keras.keras.optimizers import Adam


filename="closer_data_coords.csv"
scaler = MinMaxScaler()
model_name = "Gesture_model"



def sliding_window(X, y, frames=5):
    xs, ys = [], []

    for i in range(len(X) - frames):
        xs.append(X.iloc[i:i+frames].values)
        ys.append(y.iloc[i:i+frames])

    return np.array(xs), np.array(ys)



def load_data(filename):
    ##load data
    df = pd.read_csv(os.path.join(os.getcwd(), filename))

    # drop cols, z-coords, and nans
    df.drop(columns=['source_file', 'source_directory'], inplace=True, errors='ignore')
    df.columns = [col.replace('.', '_') for col in df.columns]
    df = df.drop(columns=[c for c in df.columns if '_z' in c], errors="ignore")
    df.dropna(inplace=True)

    # encode labels
    label_encoder = LabelEncoder()
    df['gesture_class'] = label_encoder.fit_transform(df['gesture_class'])
    num_classes = len(label_encoder.classes_)
    class_counts = df['gesture_class'].value_counts()

    # features / labels
    X_raw = df.drop(columns=['gesture_class']).apply(pd.to_numeric, errors='coerce')
    y_raw = df['gesture_class']

    X_window, y_window=sliding_window(X_raw, y_raw, frames=5)
    X_window, y_window = X_window.reshape(X_window.shape[0], -1), y_window.reshape(y_window.shape[0], -1)


    X_train, X_test, y_train, y_test = train_test_split(X_window, y_window, test_size=0.40, random_state=42, stratify=y_window)

    return X_train, X_test, y_train, y_test, num_classes, class_counts



X_train, X_test, y_train, y_test, num_classes, class_counts = load_data(filename)
## fit params
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if class_counts is not None: 
    print("Class Distribution:\n", class_counts)
    plt.figure(figsize=(8, 5))
    class_counts.plot(kind='bar', color='c')
    plt.xlabel("Gesture Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.show()

model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

if model:
    model.summary()

        
joblib.dump(scaler, f'{model_name}_scaler.pkl')
# ensure output folder exists and save scaler next to the model
if not os.path.exists('Trained-Models'):
    os.makedirs('Trained-Models')
scaler_path = os.path.join('Trained-Models', f'{model_name}_scaler.pkl')
joblib.dump(scaler, scaler_path)

model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

history = model.fit(
        X_train, y_train,
        epochs=100, batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr]
    )

# eval
y_pred = np.argmax(model.predict(X_test), axis=1)

# add new folder if not exists
if not os.path.exists('Trained-Models'):
    os.makedirs('Trained-Models')

model.save(os.path.join('Trained-Models', f'{model_name}.keras'))
# save trained model next to scaler
model.save(os.path.join('Trained-Models', f'{model_name}.keras'))
 


def print_model_accuracy(self):
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)

    # confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # normalize comfusion matrix
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 8))
    sns.heatmap(cmn, annot=True, fmt='.0%', cmap='Greens',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Normalized Confusion Matrix")
    plt.show()

def retrain_model(self, previous_model_name, new_data_file):
    ##Load new data, retrain from saved model, update scaler + weights.

    # load model
    model = load_model(os.path.join('Trained-Models', f'{previous_model_name}.keras'))
    scaler = joblib.load("gesture_model_scaler.pkl")

    scaler_path = os.path.join('Trained-Models', f'{previous_model_name}_scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    # load_new_data then refit
    df = pd.read_csv(os.path.join(os.getcwd(), filename))

    # drop cols, z-coords, and nans
    df.drop(columns=['source_file', 'source_directory'], inplace=True, errors='ignore')
    df.columns = [col.replace('.', '_') for col in df.columns]
    df = df.drop(columns=[c for c in df.columns if '_z' in c], errors="ignore")
    df.dropna(inplace=True)

    # encode labels
    label_encoder = LabelEncoder()
    df['gesture_class'] = label_encoder.fit_transform(df['gesture_class'])
    num_classes = len(label_encoder.classes_)

    # features / labels
    X = df.drop(columns=['gesture_class']).apply(pd.to_numeric, errors='coerce')
    y = df['gesture_class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    model.save(os.path.join('Trained-Models', f'{previous_model_name}.keras'))