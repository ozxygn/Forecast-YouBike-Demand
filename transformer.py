import pandas as pd
import numpy as np
import time
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models

# === CONFIGURATION ===
yb_csv           = 'youbike_status.csv'
map_csv          = 'weathermapping.csv'
weather_csv      = 'weather.csv'
target_col       = 'available_rent_bikes'
freq             = 'h'
target_date      = pd.to_datetime('2025-05-27')
look_back        = 168     # 1 week
forecast_h       = 5 * 24  # 5 days

test_station_id  = '500101022'  # Change as needed

# === Load Data ===
df_all = pd.read_csv(yb_csv, parse_dates=['mday'])
map_df = pd.read_csv(map_csv)
w_df_all = pd.read_csv(weather_csv, parse_dates=['observe_time'])

def get_weather_features_for_station(station_id, map_df, w_df_all, freq):
    m = map_df[map_df['YouBike Station ID'] == int(station_id)]
    if m.empty: return None
    w_name = m['Closest Weather Station'].iloc[0]
    w = w_df_all[w_df_all['station_name'] == w_name].set_index('observe_time')
    if w.empty: return None
    w = w[['temperature','weather']].resample(freq).agg({
        'temperature': 'mean',
        'weather': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    }).ffill()
    dummies = pd.get_dummies(w['weather'], prefix='weather')
    return pd.concat([w[['temperature']], dummies], axis=1)

def compute_kpi(y_true, y_pred, t_elapsed):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2, t_elapsed

# === Preprocessing ===
df = df_all[df_all['sno'].astype(str) == test_station_id].set_index('mday').sort_index()
ts = df[target_col].resample(freq).mean()
weather = get_weather_features_for_station(test_station_id, map_df, w_df_all, freq)

if ts.empty or weather is None or len(ts) < (look_back + forecast_h):
    raise ValueError("Insufficient data for look_back + forecast_h")

# Normalize
scaler_y = MinMaxScaler()
scaler_X = MinMaxScaler()

y_all = ts.values.reshape(-1,1)
X_feat = weather.values

y_scaled = scaler_y.fit_transform(y_all)
X_scaled = scaler_X.fit_transform(X_feat)

data_scaled = np.hstack([y_scaled, X_scaled])
time_index = weather.index

# Create sequences
X, y = [], []
for i in range(look_back, len(data_scaled) - forecast_h):
    X.append(data_scaled[i-look_back:i])
    y.append(data_scaled[i:i+forecast_h, 0])
X, y = np.array(X), np.array(y)

# Train/Test split
split = int(len(X)*0.8)
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]

# === Transformer Components ===
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(inputs.shape[-1])(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x

def build_transformer_model(input_shape, forecast_h, num_heads=4):
    inputs = layers.Input(shape=input_shape)
    x = transformer_encoder(inputs, head_size=64, num_heads=num_heads, ff_dim=128)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(forecast_h)(x)
    return models.Model(inputs, x)

# === Build, Train, Evaluate ===
model = build_transformer_model((look_back, data_scaled.shape[1]), forecast_h)
model.compile(optimizer='adam', loss='mse')

t0 = time.time()
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)
elapsed = time.time() - t0

y_pred_s = model.predict(X_test)

# Inverse scale
y_pred = np.zeros_like(y_pred_s)
y_true = np.zeros_like(y_test)
for h in range(forecast_h):
    y_pred[:, h] = scaler_y.inverse_transform(y_pred_s[:, h].reshape(-1, 1)).flatten()
    y_true[:, h] = scaler_y.inverse_transform(y_test[:, h].reshape(-1, 1)).flatten()

# Save results
metrics = []
for h in range(forecast_h):
    mae, rmse, mape, r2, _ = compute_kpi(y_true[:,h], y_pred[:,h], elapsed)
    metrics.append({
        'station_id': test_station_id,
        'horizon_hr': h+1,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Time_s': elapsed
    })

preds = []
start_idx = split + look_back
for idx in range(len(y_pred)):
    base_time = time_index[start_idx + idx]
    for h in range(forecast_h):
        preds.append({
            'station_id':  test_station_id,
            'timestamp':   base_time + timedelta(hours=h),
            'horizon_hr':  h+1,
            'actual':      y_true[idx,h],
            'prediction':  y_pred[idx,h]
        })

pd.DataFrame(metrics).to_csv('transformer_metrics.csv', index=False, float_format='%.3f')
pd.DataFrame(preds).to_csv('transformer_predictions.csv', index=False, float_format='%.2f')

print("✓ Transformer completed for station", test_station_id)
print("  • Metrics → transformer_metrics.csv")
print("  • Predictions → transformer_predictions.csv")
