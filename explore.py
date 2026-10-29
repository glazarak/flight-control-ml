import pandas as pd
import numpy as np

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