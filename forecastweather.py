import pandas as pd
import numpy as np
import time
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# === CONFIGURATION ===
yb_csv      = 'youbike_status.csv'      # YouBike data
map_csv     = 'weathermapping.csv'       # mapping station -> weather station
weather_csv = 'weather.csv'               # weather data
target_col  = 'available_rent_bikes'
num_weeks   = 3
freq        = 'h'                         # use lowercase for frequency
target_date = pd.to_datetime('2025-05-23')

# calculate window (same for all stations)
start = target_date
end   = target_date + timedelta(days=5) - timedelta(hours=1)

# === 1) Load all data ===
df_all    = pd.read_csv(yb_csv, parse_dates=['mday'])
map_df    = pd.read_csv(map_csv)
w_df_all  = pd.read_csv(weather_csv, parse_dates=['observe_time'])

# List all station IDs
station_ids = df_all['sno'].astype(str).unique()

# Helper: compute KPI
def compute_kpi(y_true, y_pred, t_elapsed):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/(y_true + 1e-5))) * 100
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2, t_elapsed

# === Function: get weather features for a station ===
def get_weather_features_for_station(station_id, map_df, w_df_all, freq):
    mapping = map_df[map_df['YouBike Station ID'] == int(station_id)]
    if mapping.empty:
        return None
    w_name = mapping['Closest Weather Station'].iloc[0]

    w_df = w_df_all[w_df_all['station_name'] == w_name].set_index('observe_time')
    if w_df.empty:
        return None

    weather_hourly = w_df[['temperature', 'weather']].resample(freq).agg({
        'temperature': 'mean',
        'weather':     lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    }).ffill()

    weather_dummies = pd.get_dummies(weather_hourly['weather'], prefix='weather')
    weather_feats = pd.concat([weather_hourly[['temperature']], weather_dummies], axis=1)

    return weather_feats, list(weather_dummies.columns)

# Containers for results
predictions_list = []
metrics_list     = []

# Loop over each station
for station_id in station_ids:
    df = df_all[df_all['sno'].astype(str) == station_id].set_index('mday').sort_index()
    ts_hourly = df[target_col].resample(freq).mean()

    if ts_hourly.empty or ts_hourly.index.min() >= start:
        continue

    # Prepare lag features
    hours_per_week = 7 * 24
    lags = [ts_hourly.shift(hours_per_week * k).rename(f'lag_{k}') for k in range(1, num_weeks+1)]
    df_lags = pd.concat(lags, axis=1)

    # Get weather features
    weather_res = get_weather_features_for_station(station_id, map_df, w_df_all, freq)
    if weather_res is None:
        continue
    weather_feats, weather_dummy_cols = weather_res

    # Features for RF & LR
    df_feats_rf = df_lags.join(weather_feats, how='inner')
    df_time = pd.DataFrame(index=df_feats_rf.index)
    df_time['dow']  = df_time.index.dayofweek
    df_time['hour'] = df_time.index.hour
    df_time = pd.get_dummies(df_time, columns=['dow','hour'], prefix=['dow','hour'])
    df_feats_lr = pd.concat([df_feats_rf, df_time], axis=1).dropna()

    # Split train/test
    mask_train = ts_hourly.index < start
    mask_test  = (ts_hourly.index >= start) & (ts_hourly.index <= end)
    y_train    = ts_hourly[mask_train]
    y_test     = ts_hourly[mask_test]

    preds = pd.DataFrame(index=y_test.index)

    # === Naive ===
    t0 = time.time()
    mean_train = y_train.mean()
    fc_naive   = pd.Series(mean_train, index=y_test.index)
    t_naive    = time.time() - t0
    preds['Naive'] = fc_naive
    m_naive = compute_kpi(y_test, fc_naive, t_naive)
    metrics_list.append((station_id, 'Naive') + m_naive)

    # --- Weekly Average ---
    t0 = time.time()
    df_eval    = df_lags.loc[start:end]
    fc_weekly  = df_eval.mean(axis=1).reindex(y_test.index)
    t_weekly   = time.time() - t0
    preds['WeeklyAvg'] = fc_weekly
    metrics_list.append((station_id, 'WeeklyAvg') + compute_kpi(y_test, fc_weekly, t_weekly))

    # --- Random Forest ---
    t0 = time.time()
    train_rf = df_feats_rf[df_feats_rf.index < start].dropna()
    test_rf  = df_feats_rf.loc[start:end].dropna()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_rf.values, ts_hourly.loc[train_rf.index])
    fc_rf = pd.Series(rf.predict(test_rf.values), index=test_rf.index)
    t_rf = time.time() - t0
    preds['RandomForest'] = fc_rf.reindex(y_test.index)
    metrics_list.append((station_id, 'RandomForest') + compute_kpi(ts_hourly.loc[test_rf.index], fc_rf, t_rf))

    # --- Linear Regression ---
    t0 = time.time()
    train_lr = df_feats_lr[df_feats_lr.index < start]
    test_lr  = df_feats_lr.loc[start:end]
    lr = LinearRegression()
    lr.fit(train_lr.values, ts_hourly.loc[train_lr.index])
    fc_lr = pd.Series(lr.predict(test_lr.values), index=test_lr.index)
    t_lr = time.time() - t0
    preds['LinearRegression'] = fc_lr.reindex(y_test.index)
    metrics_list.append((station_id, 'LinearRegression') + compute_kpi(ts_hourly.loc[test_lr.index], fc_lr, t_lr))

    # --- Prophet ---
    t0 = time.time()
    prophet_df = ts_hourly[mask_train].to_frame(name='y').join(weather_feats, how='left').dropna()
    if not prophet_df.empty:
        prophet_df = prophet_df.reset_index().rename(columns={'mday':'ds'})
        m_prophet = Prophet(daily_seasonality=True, weekly_seasonality=True)
        m_prophet.add_seasonality(name='hourly', period=24, fourier_order=5)
        m_prophet.add_regressor('temperature')
        for col in weather_dummy_cols:
            m_prophet.add_regressor(col)
        m_prophet.fit(prophet_df)
        future = weather_feats.loc[start:end].reset_index().rename(columns={'observe_time':'ds'})
        forecast = m_prophet.predict(future).set_index('ds')['yhat']
        t_prophet = time.time() - t0
        preds['Prophet'] = forecast.reindex(y_test.index)
        metrics_list.append((station_id, 'Prophet') + compute_kpi(y_test, preds['Prophet'], t_prophet))
    else:
        preds['Prophet'] = np.nan

    # --- SARIMAX ---
    t0 = time.time()
    train_ar = ts_hourly[:start - timedelta(hours=1)]
    test_ar  = ts_hourly.loc[start:end]
    if not train_ar.empty and len(test_ar) > 0:
        exog_train  = weather_feats.loc[train_ar.index][['temperature']]
        exog_future = weather_feats.loc[start:end][['temperature']]
        try:
            model_sx = SARIMAX(
                train_ar,
                order=(1,1,1),
                seasonal_order=(1,0,1,24),
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            fc_sx = model_sx.predict(start=start, end=end, exog=exog_future)
            fc_sx.index = test_ar.index
            t_sx = time.time() - t0
            preds['SARIMAX'] = fc_sx
            metrics_list.append((station_id, 'SARIMAX') + compute_kpi(test_ar, fc_sx, t_sx))
        except Exception as e:
            print(f"SARIMAX error for station {station_id}: {e}")
            preds['SARIMAX'] = np.nan
    else:
        preds['SARIMAX'] = np.nan

    # Finalize predictions for this station
    df_out = preds.reset_index().rename(columns={'index':'timestamp'})
    df_out['station_id'] = station_id
    predictions_list.append(df_out)

    print(f"Finished forecasting for station {station_id}")

# Save final results
if predictions_list:
    predictions_all = pd.concat(predictions_list, ignore_index=True)
    predictions_all.to_csv('predictions_all_stations.csv', index=False, float_format='%.2f')

if metrics_list:
    metrics_df = pd.DataFrame(
        metrics_list,
        columns=['station_id','Model','MAE','RMSE','MAPE','R2','Time_s']
    )
    metrics_df.to_csv('metrics_all_stations.csv', index=False, float_format='%.3f')

print("Forecasting process completed!\n - predictions_all_stations.csv\n - metrics_all_stations.csv")
