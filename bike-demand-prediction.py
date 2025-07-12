import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv('hour_sample.csv')  # Use the sample dataset

# Preprocessing
df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
df.set_index('datetime', inplace=True)

features = ['temp', 'atemp', 'hum', 'windspeed', 'season', 'holiday', 'workingday', 'weathersit', 'hr']
target = 'cnt'

# Normalize Features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features + [target]])

# Sequence Data for LSTM
def create_sequence(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, :-1])
        y.append(data[i+window, -1])
    return np.array(X), np.array(y)

X, y = create_sequence(data_scaled, window=24)

# Train-Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM Model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Predict and Evaluate
y_pred = model.predict(X_test)
y_pred_rescaled = y_pred * (df['cnt'].max() - df['cnt'].min()) + df['cnt'].min()
y_test_rescaled = y_test * (df['cnt'].max() - df['cnt'].min()) + df['cnt'].min()

print("MAE:", mean_absolute_error(y_test_rescaled, y_pred_rescaled))

# Plot Results
plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled[:200], label='Actual')
plt.plot(y_pred_rescaled[:200], label='Predicted')
plt.title('Hourly Bike Demand Prediction')
plt.xlabel('Hours')
plt.ylabel('Bike Count')
plt.legend()
plt.show()
