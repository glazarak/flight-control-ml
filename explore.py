import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("flight_data.csv")

print(df.shape)
print(df.head())
print(df.describe())
print(df["target_x"].nunique())

features = ["x", "y", "velocity_x", "velocity_y", "target_x", "target_y"]

X = df[features]
y = df["thrust_x"]

print(X.shape, y.shape)

df["flight"] = df.index // 500

print(df["flight"].nunique())
print(df.groupby("flight")["target_x"].nunique().unique())



rng = np.random.default_rng(42)
flight_order = rng.permutation(20)

train_flights = flight_order[:12]
val_flights = flight_order[12:16]
test_flights = flight_order[16:]

train = df[df["flight"].isin(train_flights)]
val = df[df["flight"].isin(val_flights)]
test = df[df["flight"].isin(test_flights)]

print(train.shape, val.shape, test.shape)

label = "thrust_x"

X_train = train[features]
y_train = train[label]

X_val = val[features]
y_val = val[label]

X_test = test[features]
y_test = test[label]

print(X_train.describe().loc[["min", "max", "std"]])

lin = LinearRegression()
lin.fit(X_train, y_train)

print("raw    train", mean_absolute_error(y_train, lin.predict(X_train)))
print("raw    val", mean_absolute_error(y_val, lin.predict(X_val)))

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

lin_s = LinearRegression()
lin_s.fit(X_train_s, y_train)

print("scaled train", mean_absolute_error(y_train, lin_s.predict(X_train_s)))
print("scaled val", mean_absolute_error(y_val, lin_s.predict(X_val_s)))

for name, coef in zip(features, lin_s.coef_):
    print(name, round(coef, 3))

print("intercept", round(lin.intercept_, 3))
