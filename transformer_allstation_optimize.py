import os
# === SUPPRESS TENSORFLOW WARNINGS ===
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import time
from datetime import timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow to use less memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# === CONFIGURATION ===
yb_csv           = 'youbike_status.csv'
map_csv          = 'weathermapping.csv'
weather_csv      = 'weather.csv'
target_col       = 'available_rent_bikes'
freq             = 'h'
target_date      = pd.to_datetime('2025-05-27')
look_back        = 168     # 168 hours = 1 week of history
forecast_h       = 5 * 24  # 5 days (120-hour forecast)

# Optimized training parameters
MAX_EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 5
MIN_DELTA = 0.001
VALIDATION_SPLIT = 0.2

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
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]
    
    if len(y_true_f) == 0:
        return np.nan, np.nan, np.nan, np.nan, t_elapsed
        
    mae  = mean_absolute_error(y_true_f, y_pred_f)
    rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
    mape = np.mean(np.abs((y_true_f - y_pred_f) / (y_true_f + 1e-5))) * 100
    r2   = r2_score(y_true_f, y_pred_f)
    return mae, rmse, mape, r2, t_elapsed

def transformer_encoder(inputs, head_size=32, num_heads=2, ff_dim=64, dropout=0.2):
    """Optimized transformer encoder - smaller but efficient"""
    # Multi-head attention
    x = layers.MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads, 
        dropout=dropout,
        use_bias=False  # Reduce parameters
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    # Feed forward
    ff = layers.Dense(ff_dim, activation='relu', use_bias=False)(x)
    ff = layers.Dense(inputs.shape[-1], use_bias=False)(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x

def build_optimized_transformer_model(input_shape, forecast_h):
    """Optimized transformer model with fewer parameters"""
    inputs = layers.Input(shape=input_shape)
    
    # Single transformer encoder layer (instead of multiple)
    x = transformer_encoder(inputs, head_size=32, num_heads=2, ff_dim=64, dropout=0.2)
    
    # More efficient pooling and dense layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu', use_bias=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu', use_bias=False)(x)
    outputs = layers.Dense(forecast_h, use_bias=False)(x)
    
    model = models.Model(inputs, outputs)
    
    # Use Adam with lower learning rate for stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def prepare_data_efficiently(df, weather, look_back, forecast_h):
    """Efficient data preparation with vectorized operations"""
    # Scale data
    scaler_y = MinMaxScaler()
    y_all = df.values.reshape(-1, 1)
    y_scaled = scaler_y.fit_transform(y_all)

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(weather.values)

    data_scaled = np.hstack([y_scaled, X_scaled])
    
    # Vectorized sequence creation
    n_samples = len(data_scaled) - look_back - forecast_h + 1
    if n_samples <= 0:
        return None, None, None, None, None
    
    # Create sequences using numpy array operations
    X_seq = np.zeros((n_samples, look_back, data_scaled.shape[1]))
    y_seq = np.zeros((n_samples, forecast_h))
    
    valid_indices = []
    for i in range(n_samples):
        seq_x = data_scaled[i:i + look_back]
        seq_y = data_scaled[i + look_back:i + look_back + forecast_h, 0]
        
        if not (np.isnan(seq_x).any() or np.isnan(seq_y).any()):
            X_seq[len(valid_indices)] = seq_x
            y_seq[len(valid_indices)] = seq_y
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        return None, None, None, None, None
    
    X_seq = X_seq[:len(valid_indices)]
    y_seq = y_seq[:len(valid_indices)]
    
    return X_seq, y_seq, scaler_y, weather.index, valid_indices

def save_intermediate_results(results, predictions, batch_num):
    """Save intermediate results to prevent data loss"""
    if results:
        pd.DataFrame(results, columns=['station_id','horizon_hr','MAE','RMSE','MAPE','R2','Time_s']) \
            .to_csv(f'transformer_metrics_batch_{batch_num}.csv', index=False)
    
    if predictions:
        pd.DataFrame(predictions) \
            .to_csv(f'transformer_predictions_batch_{batch_num}.csv', index=False, float_format='%.2f')

# === Load Data Once ===
print("Loading data...")
df_all = pd.read_csv(yb_csv, parse_dates=['mday'])
map_df = pd.read_csv(map_csv)
w_df_all = pd.read_csv(weather_csv, parse_dates=['observe_time'])

# Containers for aggregated results
results = []
predictions_list = []
station_ids = df_all['sno'].astype(str).unique()

print(f"Processing {len(station_ids)} stations with optimized transformer...")

# Processing parameters
BATCH_SIZE_STATIONS = 100  # Process and save every 100 stations
processed_count = 0
start_time = time.time()

# Main loop with optimizations
for i, station_id in enumerate(station_ids):
    try:
        station_start_time = time.time()
        
        # 1) Prepare time series for station
        df = df_all[df_all['sno'].astype(str) == station_id].set_index('mday').sort_index()
        ts = df[target_col].resample(freq).mean().bfill().ffill()

        # 2) Extract weather features
        weather = get_weather_features_for_station(station_id, map_df, w_df_all, freq)
        if ts.empty or weather is None or len(ts) < (look_back + forecast_h):
            print(f"✗ Skipped station {station_id} - insufficient data")
            continue

        # 3) Prepare data efficiently
        data_prep = prepare_data_efficiently(ts, weather, look_back, forecast_h)
        if data_prep[0] is None:
            print(f"✗ Skipped station {station_id} - data preparation failed")
            continue
        
        X_seq, y_seq, scaler_y, time_idx, valid_indices = data_prep

        # 4) Train-test split
        split = int(len(X_seq) * 0.8)
        X_train, y_train = X_seq[:split], y_seq[:split]
        X_test, y_test = X_seq[split:], y_seq[split:]

        # 5) Build transformer model
        model = build_optimized_transformer_model((look_back, X_seq.shape[2]), forecast_h)

        # 6) Setup callbacks for faster training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                min_delta=MIN_DELTA,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0
            )
        ]

        # 7) Train model
        t0 = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0
        )
        elapsed = time.time() - t0

        # 8) Predict and inverse scale
        y_pred_s = model.predict(X_test, verbose=0)
        y_pred = np.zeros_like(y_pred_s)
        y_true = np.zeros_like(y_test)
        
        for h in range(forecast_h):
            y_pred[:, h] = scaler_y.inverse_transform(y_pred_s[:, h].reshape(-1, 1)).flatten()
            y_true[:, h] = scaler_y.inverse_transform(y_test[:, h].reshape(-1, 1)).flatten()

        # 9) Collect predictions
        base_idx = split + look_back
        for idx in range(len(y_pred)):
            base_time = time_idx[base_idx + valid_indices[idx]]
            for h in range(forecast_h):
                predictions_list.append({
                    'station_id': station_id,
                    'timestamp': base_time + timedelta(hours=h),
                    'horizon_hr': h+1,
                    'actual': y_true[idx, h],
                    'prediction': y_pred[idx, h]
                })

        # 10) Compute and store KPIs
        for h in range(forecast_h):
            mae, rmse, mape, r2, _ = compute_kpi(y_true[:, h], y_pred[:, h], elapsed)
            results.append([station_id, h+1, mae, rmse, mape, r2, elapsed])

        processed_count += 1
        station_elapsed = time.time() - station_start_time
        
        # Progress reporting
        if processed_count % 10 == 0:
            avg_time_per_station = station_elapsed
            remaining_stations = len(station_ids) - processed_count
            estimated_remaining_time = remaining_stations * avg_time_per_station / 3600  # in hours
            
            print(f"✓ Completed {processed_count}/{len(station_ids)} stations")
            print(f"  Last station: {station_id} ({station_elapsed:.1f}s)")
            print(f"  Estimated remaining time: {estimated_remaining_time:.1f} hours")

        # Save intermediate results every BATCH_SIZE_STATIONS
        if processed_count % BATCH_SIZE_STATIONS == 0:
            batch_num = processed_count // BATCH_SIZE_STATIONS
            print(f"Saving intermediate results (batch {batch_num})...")
            save_intermediate_results(results, predictions_list, batch_num)

        # Clean up memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    except Exception as e:
        print(f"✗ Error processing station {station_id}: {str(e)}")
        continue

# === Save Final Results ===
print("\nSaving final results...")
pd.DataFrame(results, columns=['station_id','horizon_hr','MAE','RMSE','MAPE','R2','Time_s']) \
    .to_csv('transformer_metrics_all_stations.csv', index=False)

pd.DataFrame(predictions_list) \
    .to_csv('transformer_predictions_all_stations.csv', index=False, float_format='%.2f')

total_time = time.time() - start_time
print("✓ All done.")
print(f"  • Total processing time: {total_time/3600:.2f} hours")
print(f"  • Successfully processed: {processed_count}/{len(station_ids)} stations")
print(f"  • Average time per station: {total_time/processed_count:.1f} seconds")
print("  • Metrics → transformer_metrics_all_stations.csv")
print("  • Predictions → transformer_predictions_all_stations.csv")