import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def fit_arima(train, test):
    print("\n── ARIMA(2,1,2) ──")
    model = SARIMAX(
        train,
        order=(2, 1, 2),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    pred = fit.forecast(steps=len(test))
    pred.index = test.index
    print(f"   AIC : {fit.aic:.2f}")
    return pred


def fit_sarima(train, test):
    print("\n── SARIMA(2,1,2)(1,1,1,7) ──")
    model = SARIMAX(
        train,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    pred = fit.forecast(steps=len(test))
    pred.index = test.index
    print(f"   AIC : {fit.aic:.2f}")
    return pred


def fit_prophet(train, test):
    print("\n── Prophet ──")
    df_train = train.reset_index().rename(
        columns={"date": "ds", "consommation_kwh": "y"}
    )
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1,
    )
    m.fit(df_train)
    future = m.make_future_dataframe(periods=len(test))
    forecast = m.predict(future)
    pred = forecast.set_index("ds")["yhat"].loc[test.index]
    return pred


def build_lstm_sequences(series: np.ndarray, window: int = 30):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def fit_lstm(train, test, window: int = 30):
    print("\n── LSTM ──")
    scaler = MinMaxScaler()
    all_data = np.concatenate([train.values, test.values]).reshape(-1, 1)
    scaled = scaler.fit_transform(all_data)

    train_sc = scaled[: len(train)]
    X_tr, y_tr = build_lstm_sequences(train_sc.flatten(), window)
    X_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)

    tf.random.set_seed(42)
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(window, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
    model.fit(
        X_tr,
        y_tr,
        epochs=80,
        batch_size=32,
        validation_split=0.1,
        callbacks=[es],
        verbose=0,
    )

    # Prédiction pas-à-pas sur le test
    input_seq = list(train_sc.flatten()[-window:])
    preds_sc = []
    for _ in range(len(test)):
        x = np.array(input_seq[-window:]).reshape(1, window, 1)
        p = model.predict(x, verbose=0)[0, 0]
        preds_sc.append(p)
        input_seq.append(p)

    preds = scaler.inverse_transform(np.array(preds_sc).reshape(-1, 1)).flatten()
    return pd.Series(preds, index=test.index)
