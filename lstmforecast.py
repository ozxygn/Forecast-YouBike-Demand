import os
# === SUPPRESS TENSORFLOW WARNINGS ===
# Disable oneDNN custom operations warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import time
from datetime import timedelta
# Suppress TensorFlow info and warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense

# === CONFIGURATION ===
yb_csv      = 'youbike_status.csv'
map_csv     = 'weathermapping.csv'
weather_csv = 'weather.csv'
target_col  = 'available_rent_bikes'
freq        = 'h'
target_date = pd.to_datetime('2025-05-27')

look_back   = 168     # 168 hours = 1 week of history
forecast_h  = 5 * 24  # 5 days (120-hour forecast)

# Load data
df_all    = pd.read_csv(yb_csv, parse_dates=['mday'])
map_df    = pd.read_csv(map_csv)
w_df_all  = pd.read_csv(weather_csv, parse_dates=['observe_time'])

# Function to extract weather features for a station
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
    weather_dummies = pd.get_dummies(w_df['weather'], prefix='weather')
    return pd.concat([w_df[['temperature']], weather_dummies], axis=1)

# KPI function
def compute_kpi(y_true, y_pred, t_elapsed):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]
    mae  = mean_absolute_error(y_true_f, y_pred_f)
    rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
    mape = np.mean(np.abs((y_true_f - y_pred_f) / (y_true_f + 1e-5))) * 100
    r2   = r2_score(y_true_f, y_pred_f)
    return mae, rmse, mape, r2, t_elapsed

# Container for results
results = []
predictions_list = []
station_ids = df_all['sno'].astype(str).unique()
start = target_date
end   = target_date + timedelta(hours=forecast_h - 1)

for station_id in station_ids:
    # 1) Prepare YouBike time series
    df = df_all[df_all['sno'].astype(str) == station_id].set_index('mday').sort_index()
    ts = df[target_col].resample(freq).mean().bfill().ffill()

    # 2) Extract weather features
    weather = get_weather_features_for_station(station_id, map_df, w_df_all, freq)
    if ts.empty or weather is None or len(ts) < (look_back + forecast_h):
        continue

    # 3) Separate scaling
    scaler_y = MinMaxScaler()
    y_all = ts.values.reshape(-1, 1)
    y_scaled = scaler_y.fit_transform(y_all)

    scaler_X = MinMaxScaler()
    X_feat = weather.values
    X_scaled = scaler_X.fit_transform(X_feat)

    # 4) Combine and create sequence dataset
    data_scaled = np.hstack([y_scaled, X_scaled])
    time_index = weather.index
    X, y = [], []
    for i in range(look_back, len(data_scaled) - forecast_h):
        seq_x = data_scaled[i - look_back:i]
        seq_y = data_scaled[i:i + forecast_h, 0]
        if np.isnan(seq_x).any() or np.isnan(seq_y).any():
            continue
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X); y = np.array(y)
    if len(X) == 0:
        continue

    # 5) Split into train/test sets
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    # 6) Define model with Input layer
    model = Sequential([
        Input(shape=(look_back, data_scaled.shape[1])),
        LSTM(50),
        Dense(forecast_h)
    ])
    model.compile(optimizer='adam', loss='mse')

    # 7) Training
    t0 = time.time()
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    elapsed = time.time() - t0

    # 8) tf.function for prediction to reduce retracing
    @tf.function(reduce_retracing=True)
    def predict_fn(x):
        return model(x, training=False)

    # 9) Predict & inverse-scale only the target
    y_pred_s = predict_fn(tf.constant(X_test))
    y_pred_s = y_pred_s.numpy()
    y_pred = np.zeros_like(y_pred_s)
    y_true = np.zeros_like(y_test)
    for h in range(forecast_h):
        y_pred[:, h] = scaler_y.inverse_transform(y_pred_s[:, h].reshape(-1, 1)).flatten()
        y_true[:, h] = scaler_y.inverse_transform(y_test[:, h].reshape(-1, 1)).flatten()

    # 10) Save predictions with timestamp
    base_idx = split + look_back
    for idx in range(len(y_pred)):
        base_time = time_index[base_idx + idx]
        for h in range(forecast_h):
            predictions_list.append({
                'station_id': station_id,
                'timestamp':   base_time + timedelta(hours=h),
                'horizon_hr':  h + 1,
                'actual':      y_true[idx, h],
                'prediction':  y_pred[idx, h]
            })

    # 11) Save metrics per horizon
    for h in range(forecast_h):
        mae, rmse, mape, r2, _ = compute_kpi(y_true[:, h], y_pred[:, h], elapsed)
        results.append([station_id, h + 1, mae, rmse, mape, r2, elapsed])

    print(f"✓ Completed LSTM (look_back={look_back}) for station {station_id}")

# 12) Save results to CSV
pd.DataFrame(
    results,
    columns=['station_id','horizon_hr','MAE','RMSE','MAPE','R2','Time_s']
).to_csv('lstm_metrics_all_stations.csv', index=False)

pd.DataFrame(predictions_list).to_csv(
    'lstm_predictions_all_stations.csv',
    index=False, float_format='%.2f'
)

print("✓ All done.")
print("  • Metrics → lstm_metrics_all_stations.csv")
print("  • Predictions → lstm_predictions_all_stations.csv")
