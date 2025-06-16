import pandas as pd
import numpy as np
import time
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

# === CONFIGURATION ===
yb_csv      = 'youbike_status.csv'      # YouBike data
map_csv     = 'weathermapping.csv'       # mapping station -> weather station
weather_csv = 'weather.csv'               # weather data
target_col  = 'available_rent_bikes'
num_weeks   = 3
freq        = 'h'                         # use lowercase for frequency
target_date = pd.to_datetime('2025-05-27')

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
    # Find YouBike to Weather Station mapping
    mapping = map_df[map_df['YouBike Station ID'] == int(station_id)]
    if mapping.empty:
        return None
    w_name = mapping['Closest Weather Station'].iloc[0]

    # Filter weather data by weather station name
    w_df = w_df_all[w_df_all['station_name'] == w_name].set_index('observe_time')
    if w_df.empty:
        return None

    # Resample hourly (frequency 'h'): mean temperature + mode of weather category
    weather_hourly = w_df[['temperature', 'weather']].resample(freq).agg({
        'temperature': 'mean',
        'weather':     lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    })

    # Fill NaN in both columns using forward fill
    weather_hourly = weather_hourly.ffill()

    # One-hot encode weather category
    weather_dummies = pd.get_dummies(weather_hourly['weather'], prefix='weather')
    weather_feats = pd.concat([weather_hourly[['temperature']], weather_dummies], axis=1)

    # Return weather feature DataFrame and list of weather dummy columns
    return weather_feats, list(weather_dummies.columns)

# Container to store all predictions and metrics
predictions_list = []
metrics_list     = []

# Loop over each station
for station_id in station_ids:
    # === 2) Prepare ts_hourly (YouBike) for this station ===
    df = df_all[df_all['sno'].astype(str) == station_id].set_index('mday').sort_index()
    ts_hourly = df[target_col].resample(freq).mean()

    # If data is insufficient for the forecast window, skip the station
    if ts_hourly.empty or ts_hourly.index.min() >= start:
        continue

    # === 3) Prepare lag features (1–num_weeks) ===
    hours_per_week = 7 * 24
    lags = []
    for k in range(1, num_weeks + 1):
        lags.append(ts_hourly.shift(periods=hours_per_week * k).rename(f'lag_{k}'))
    df_lags = pd.concat(lags, axis=1)

    # === 4) Get weather features using separate function ===
    weather_res = get_weather_features_for_station(station_id, map_df, w_df_all, freq)
    if weather_res is None:
        continue
    weather_feats, weather_dummy_cols = weather_res

    # === 5) Combine features for Random Forest and Linear Regression ===
    df_feats_rf = df_lags.join(weather_feats, how='inner')

    # For Linear Regression, add dow & hour dummies
    df_time = pd.DataFrame(index=df_feats_rf.index)
    df_time['dow']  = df_time.index.dayofweek
    df_time['hour'] = df_time.index.hour
    df_time = pd.get_dummies(df_time, columns=['dow','hour'], prefix=['dow','hour'])
    df_feats_lr = pd.concat([df_feats_rf, df_time], axis=1).dropna()

    # === 6) Split train/test ===
    mask_train = ts_hourly.index < start
    mask_test  = (ts_hourly.index >= start) & (ts_hourly.index <= end)
    y_train    = ts_hourly[mask_train]
    y_test     = ts_hourly[mask_test]

    # Prediction container for this station
    preds = pd.DataFrame(index=y_test.index)

    # === Naive ===
    t0 = time.time()
    mean_train = y_train.mean()
    fc_naive   = pd.Series(mean_train, index=y_test.index)
    t_naive    = time.time() - t0
    preds['Naive'] = fc_naive
    m_naive = compute_kpi(y_test, fc_naive, t_naive)
    metrics_list.append((station_id, 'Naive') + m_naive)

    # === Weekly Average ===
    t0 = time.time()
    df_eval   = df_lags.loc[start:end]
    fc_weekly = df_eval.mean(axis=1)
    t_weekly  = time.time() - t0
    fc_weekly = fc_weekly.reindex(y_test.index)
    preds['WeeklyAvg'] = fc_weekly
    m_weekly = compute_kpi(y_test, fc_weekly, t_weekly)
    metrics_list.append((station_id, 'WeeklyAvg') + m_weekly)

    # === Random Forest ===
    t0 = time.time()
    train_rf   = df_feats_rf[df_feats_rf.index < start].dropna()
    test_rf    = df_feats_rf.loc[start:end].dropna()
    X_train_rf = train_rf.values
    X_test_rf  = test_rf.values
    y_train_rf = ts_hourly.loc[train_rf.index]
    y_test_rf  = ts_hourly.loc[test_rf.index]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_rf, y_train_rf)
    fc_rf = pd.Series(rf.predict(X_test_rf), index=y_test_rf.index)
    t_rf = time.time() - t0
    preds['RandomForest'] = preds.index.map(fc_rf)
    m_rf = compute_kpi(y_test_rf, fc_rf, t_rf)
    metrics_list.append((station_id, 'RandomForest') + m_rf)

    # === Linear Regression ===
    t0 = time.time()
    train_lr     = df_feats_lr[df_feats_lr.index < start]
    test_lr      = df_feats_lr.loc[start:end]
    X_train_lr   = train_lr.values
    X_test_lr    = test_lr.values
    y_train_lr   = ts_hourly.loc[train_lr.index]
    y_test_lr    = ts_hourly.loc[test_lr.index]
    lr = LinearRegression()
    lr.fit(X_train_lr, y_train_lr)
    fc_lr = pd.Series(lr.predict(X_test_lr), index=y_test_lr.index)
    t_lr = time.time() - t0
    preds['LinearRegression'] = preds.index.map(fc_lr)
    m_lr = compute_kpi(y_test_lr, fc_lr, t_lr)
    metrics_list.append((station_id, 'LinearRegression') + m_lr)

    # === Prophet ===
    t0 = time.time()
    prophet_df = ts_hourly[mask_train].to_frame(name='y').join(weather_feats, how='left').dropna()
    if not prophet_df.empty:
        prophet_df = prophet_df.reset_index().rename(columns={'mday': 'ds'})
        m_prophet = Prophet(daily_seasonality=True, weekly_seasonality=True)
        m_prophet.add_seasonality(name='hourly', period=24, fourier_order=5)
        m_prophet.add_regressor('temperature')
        for col in weather_dummy_cols:
            m_prophet.add_regressor(col)
        m_prophet.fit(prophet_df)

        future = weather_feats.loc[start:end].reset_index().rename(columns={'observe_time':'ds'})
        forecast = m_prophet.predict(future).set_index('ds')['yhat']
        t_prophet = time.time() - t0
        forecast = forecast.reindex(y_test.index)
        preds['Prophet'] = forecast
        m_prop = compute_kpi(y_test, forecast, t_prophet)
        metrics_list.append((station_id, 'Prophet') + m_prop)
    else:
        preds['Prophet'] = np.nan

    # === ARIMA ===
    t0 = time.time()
    train_ar = ts_hourly[:start - timedelta(hours=1)]
    test_ar  = ts_hourly.loc[start:end]
    if not train_ar.empty and len(test_ar) > 0:
        try:
            model_ar = ARIMA(train_ar, order=(1,1,1)).fit()
            fc_ar = model_ar.forecast(steps=len(test_ar))
            fc_ar.index = test_ar.index
            t_ar = time.time() - t0
            preds['ARIMA'] = fc_ar
            m_ar = compute_kpi(test_ar, fc_ar, t_ar)
            metrics_list.append((station_id, 'ARIMA') + m_ar)
        except:
            preds['ARIMA'] = np.nan
    else:
        preds['ARIMA'] = np.nan

    # Add Actual and station_id columns
    preds.insert(0, 'Actual', y_test)
    preds = preds.reset_index().rename(columns={'index': 'timestamp'})
    preds['station_id'] = station_id
    predictions_list.append(preds)

    print(f"Finished forecasting for station {station_id}")

# === 7) Save final results ===
if predictions_list:
    predictions_all = pd.concat(predictions_list, ignore_index=True)
    predictions_all.to_csv('predictions_all_stations.csv', index=False, float_format='%.2f')

if metrics_list:
    metrics_df = pd.DataFrame(
        metrics_list,
        columns=['station_id','Model','MAE','RMSE','MAPE','R2','Time_s']
    )
    metrics_df.to_csv('metrics_all_stations.csv', index=False, float_format='%.3f')

print("Forecasting process for all stations completed!\n - predictions_all_stations.csv\n - metrics_all_stations.csv")
