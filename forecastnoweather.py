import pandas as pd
import numpy as np
import time
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

def main():
    # === CONFIGURATION ===
    yb_csv      = 'youbike_status.csv'      # YouBike data file
    target_col  = 'available_rent_bikes'
    num_weeks   = 3
    freq        = 'h'                    # use lowercase for frequency
    target_date = pd.to_datetime('2025-05-27')

    # compute window (same for all stations)
    start = target_date
    end   = target_date + timedelta(days=5) - timedelta(hours=1)

    # === 1) Load all data ===
    df_all    = pd.read_csv(yb_csv, parse_dates=['mday'])

    # List all station IDs
    station_ids = df_all['sno'].astype(str).unique()

    # Helper: compute KPIs
    def compute_kpi(y_true, y_pred, t_elapsed):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred)/(y_true + 1e-5))) * 100
        r2   = r2_score(y_true, y_pred)
        return mae, rmse, mape, r2, t_elapsed

    # Containers for storing all predictions and metrics
    predictions_list = []
    metrics_list     = []

    # Loop through each station
    for station_id in station_ids:
        # === 2) Prepare hourly time series for this station ===
        df = df_all[df_all['sno'].astype(str) == station_id].set_index('mday').sort_index()
        ts_hourly = df[target_col].resample(freq).mean()

        # If there's not enough data for the forecast window, skip this station
        if ts_hourly.empty or ts_hourly.index.min() >= start:
            continue

        # === 3) Create lag features (1–num_weeks) ===
        hours_per_week = 7 * 24
        lags = []
        for k in range(1, num_weeks + 1):
            lags.append(ts_hourly.shift(periods=hours_per_week * k).rename(f'lag_{k}'))
        df_lags = pd.concat(lags, axis=1)

        # Remove rows with NaN (due to lags)
        df_lags = df_lags.dropna()

        # If df_lags is empty after dropping NaNs, skip station
        if df_lags.empty:
            continue

        # === 4) Combine features for Random Forest and Linear Regression (no weather) ===
        df_feats_rf = df_lags.copy()

        # For Linear Regression, add day-of-week & hour dummy variables
        df_time = pd.DataFrame(index=df_feats_rf.index)
        df_time['dow']  = df_time.index.dayofweek
        df_time['hour'] = df_time.index.hour
        df_time = pd.get_dummies(df_time, columns=['dow','hour'], prefix=['dow','hour'])
        df_feats_lr = pd.concat([df_feats_rf, df_time], axis=1)

        # === 5) Split train/test ===
        mask_train = ts_hourly.index < start
        mask_test  = (ts_hourly.index >= start) & (ts_hourly.index <= end)
        y_train    = ts_hourly[mask_train]
        y_test     = ts_hourly[mask_test]

        # If there's no test data, skip station
        if y_test.empty:
            continue

        # Container for predictions for this station
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
        fc_weekly_full = df_eval.mean(axis=1).reindex(y_test.index)
        t_weekly  = time.time() - t0
        preds['WeeklyAvg'] = fc_weekly_full
        # Only compute KPI on non-NaN indices
        valid_weekly = fc_weekly_full.dropna().index
        if not valid_weekly.empty:
            m_weekly = compute_kpi(y_test.loc[valid_weekly], fc_weekly_full.loc[valid_weekly], t_weekly)
        else:
            m_weekly = (np.nan, np.nan, np.nan, np.nan, t_weekly)
        metrics_list.append((station_id, 'WeeklyAvg') + m_weekly)

        # === Random Forest ===
        t0 = time.time()
        train_rf   = df_feats_rf[df_feats_rf.index < start]
        test_rf    = df_feats_rf.loc[start:end]

        # If train or test sets are empty, skip Random Forest
        if train_rf.empty or test_rf.empty:
            preds['RandomForest'] = np.nan
            metrics_list.append((station_id, 'RandomForest', np.nan, np.nan, np.nan, np.nan, 0.0))
        else:
            X_train_rf = train_rf.values
            X_test_rf  = test_rf.values
            y_train_rf = ts_hourly.loc[train_rf.index]
            y_test_rf  = ts_hourly.loc[test_rf.index]

            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_rf, y_train_rf)
            fc_rf = pd.Series(rf.predict(X_test_rf), index=y_test_rf.index)
            t_rf = time.time() - t0

            preds['RandomForest'] = fc_rf
            m_rf = compute_kpi(y_test_rf, fc_rf, t_rf)
            metrics_list.append((station_id, 'RandomForest') + m_rf)

        # === Linear Regression ===
        t0 = time.time()
        train_lr = df_feats_lr[df_feats_lr.index < start]
        test_lr  = df_feats_lr.loc[start:end]

        # If train or test sets are empty, skip Linear Regression
        if train_lr.empty or test_lr.empty:
            preds['LinearRegression'] = np.nan
            metrics_list.append((station_id, 'LinearRegression', np.nan, np.nan, np.nan, np.nan, 0.0))
        else:
            X_train_lr = train_lr.values
            X_test_lr  = test_lr.values
            y_train_lr = ts_hourly.loc[train_lr.index]
            y_test_lr  = ts_hourly.loc[test_lr.index]

            lr = LinearRegression()
            lr.fit(X_train_lr, y_train_lr)
            fc_lr = pd.Series(lr.predict(X_test_lr), index=y_test_lr.index)
            t_lr = time.time() - t0

            preds['LinearRegression'] = fc_lr
            m_lr = compute_kpi(y_test_lr, fc_lr, t_lr)
            metrics_list.append((station_id, 'LinearRegression') + m_lr)

        # === Prophet (no regressors) ===
        t0 = time.time()
        prophet_df = ts_hourly[mask_train].to_frame(name='y').reset_index().rename(columns={'mday':'ds'})
        if not prophet_df.empty:
            m_prophet = Prophet(daily_seasonality=True, weekly_seasonality=True)
            m_prophet.add_seasonality(name='hourly', period=24, fourier_order=5)
            m_prophet.fit(prophet_df)

            future = pd.DataFrame({'ds': y_test.index})
            forecast = m_prophet.predict(future).set_index('ds')['yhat']
            t_prophet = time.time() - t0
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

    # === 6) Save final results ===
    if predictions_list:
        predictions_all = pd.concat(predictions_list, ignore_index=True)
        predictions_all.to_csv('predictions_all_stations_no_weather.csv', index=False, float_format='%.2f')

    if metrics_list:
        metrics_df = pd.DataFrame(
            metrics_list,
            columns=['station_id','Model','MAE','RMSE','MAPE','R2','Time_s']
        )
        metrics_df.to_csv('metrics_all_stations_no_weather.csv', index=False, float_format='%.3f')

    print(
        "Forecast process without weather data for all stations completed!\n"
        " - predictions_all_stations_no_weather.csv\n"
        " - metrics_all_stations_no_weather.csv"
    )

if __name__ == "__main__":
    main()
# This script forecasts YouBike station bike availability without weather data.
# It uses various models including Naive, Weekly Average, Random Forest, Linear Regression, Prophet, and ARIMA.