import os
import logging

# === SUPPRESS TENSORFLOW & OS-LEVEL LOGS ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'       # FATAL only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import time
from datetime import timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers, models

# === CONFIGURATION ===
yb_csv      = 'youbike_status.csv'
map_csv     = 'weathermapping.csv'
weather_csv = 'weather.csv'
target_col  = 'available_rent_bikes'
freq        = 'h'
look_back   = 168      # 168 hours = 1 week
forecast_h  = 5 * 24   # 5 days (120-hour forecast)

# === Helper Functions ===
def get_weather_features_for_station(station_id, map_df, w_df_all, freq):
    mapping = map_df[map_df['YouBike Station ID'] == int(station_id)]
    if mapping.empty:
        return None
    w_name = mapping['Closest Weather Station'].iloc[0]
    w_df = w_df_all[w_df_all['station_name'] == w_name].set_index('observe_time')
    if w_df.empty:
        return None
    w_df = w_df[['temperature', 'weather']].resample(freq).agg({
        'temperature': 'mean',
        'weather': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    }).bfill().ffill()
    dummies = pd.get_dummies(w_df['weather'], prefix='weather')
    return pd.concat([w_df[['temperature']], dummies], axis=1)

def compute_kpi(y_true, y_pred, t_elapsed):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    mae  = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-5))) * 100
    r2   = r2_score(y_t, y_p)
    return mae, rmse, mape, r2, t_elapsed

def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(inputs.shape[-1])(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x

def build_transformer_model(input_shape, forecast_h):
    inputs = layers.Input(shape=input_shape)
    x = transformer_encoder(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(forecast_h)(x)
    return models.Model(inputs, outputs)


# === Load Data ===
df_all   = pd.read_csv(yb_csv, parse_dates=['mday'])
map_df   = pd.read_csv(map_csv)
w_df_all = pd.read_csv(weather_csv, parse_dates=['observe_time'])

results = []
predictions_list = []
station_ids = df_all['sno'].astype(str).unique()

# === Main Loop ===
for station_id in station_ids:
    # 1) Prepare time series
    df = df_all[df_all['sno'].astype(str) == station_id].set_index('mday').sort_index()
    ts = df[target_col].resample(freq).mean().bfill().ffill()

    # 2) Weather features
    weather = get_weather_features_for_station(station_id, map_df, w_df_all, freq)
    if ts.empty or weather is None or len(ts) < (look_back + forecast_h):
        continue

    # 3) Scaling
    scaler_y = MinMaxScaler()
    y_all = ts.values.reshape(-1, 1)
    y_scaled = scaler_y.fit_transform(y_all)

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(weather.values)

    # 4) Sequence building
    data_scaled = np.hstack([y_scaled, X_scaled])
    time_idx = weather.index
    X_seq, y_seq = [], []
    for i in range(look_back, len(data_scaled) - forecast_h):
        seq_x = data_scaled[i - look_back:i]
        seq_y = data_scaled[i:i + forecast_h, 0]
        if np.isnan(seq_x).any() or np.isnan(seq_y).any():
            continue
        X_seq.append(seq_x)
        y_seq.append(seq_y)
    if not X_seq:
        continue
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # 5) Train-test split
    split = int(len(X_seq) * 0.8)
    X_train, y_train = X_seq[:split], y_seq[:split]
    X_test, y_test   = X_seq[split:], y_seq[split:]

    # 6) Build & compile model
    model = build_transformer_model((look_back, data_scaled.shape[1]), forecast_h)
    model.compile(optimizer='adam', loss='mse')

    # — Pre-trace predict graph once to avoid retracing warnings — 
    dummy = np.zeros((1, look_back, data_scaled.shape[1]), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)

    # 7) Train
    t0 = time.time()
    model.fit(X_train, y_train, epochs=3, batch_size=16, verbose=0)
    elapsed = time.time() - t0

    # 8) Predict & inverse scale
    y_pred_s = model.predict(X_test, verbose=0)
    y_pred = np.zeros_like(y_pred_s)
    y_true = np.zeros_like(y_test)
    for h in range(forecast_h):
        y_pred[:, h] = scaler_y.inverse_transform(y_pred_s[:, h].reshape(-1, 1)).flatten()
        y_true[:, h] = scaler_y.inverse_transform(y_test[:, h].reshape(-1, 1)).flatten()

    # 9) Collect predictions
    base_idx = split + look_back
    for idx in range(len(y_pred)):
        base_time = time_idx[base_idx + idx]
        for h in range(forecast_h):
            predictions_list.append({
                'station_id': station_id,
                'timestamp':   base_time + timedelta(hours=h),
                'horizon_hr':  h+1,
                'actual':      y_true[idx, h],
                'prediction':  y_pred[idx, h]
            })

    # 10) Compute KPIs
    for h in range(forecast_h):
        mae, rmse, mape, r2, _ = compute_kpi(y_true[:, h], y_pred[:, h], elapsed)
        results.append([station_id, h+1, mae, rmse, mape, r2, elapsed])

    print(f"✓ Transformer completed for station {station_id}")

# === Save Outputs ===
pd.DataFrame(results, columns=['station_id','horizon_hr','MAE','RMSE','MAPE','R2','Time_s']) \
  .to_csv('transformer_metrics_all_stations.csv', index=False)

pd.DataFrame(predictions_list) \
  .to_csv('transformer_predictions_all_stations.csv', index=False, float_format='%.2f')

print("✓ All done.")
print("  • Metrics → transformer_metrics_all_stations.csv")
print("  • Predictions → transformer_predictions_all_stations.csv")
