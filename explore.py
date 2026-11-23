import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("flight_data.csv")

print(df.shape)
print(df.head())
print(df.describe())
print(df["target_x"].nunique())

features = ["x", "y", "velocity_x", "velocity_y", "target_x", "target_y"]


df["flight"] = df.index // 500

print(df["flight"].nunique()) # how many distinct flights
print(df.groupby("flight")["target_x"].nunique().unique()) # how many targets in each flight (should be 1 for each)



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

# Linear Regression

lin = LinearRegression()
lin.fit(X_train, y_train) # trains the model with the training flights only

print("raw    train", mean_absolute_error(y_train, lin.predict(X_train)))
print("raw    val", mean_absolute_error(y_val, lin.predict(X_val)))

# Linear Regression with Standard Scaling

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

lin_s = LinearRegression()
lin_s.fit(X_train_s, y_train)

print("scaled train", mean_absolute_error(y_train, lin_s.predict(X_train_s)))
print("scaled val", mean_absolute_error(y_val, lin_s.predict(X_val_s)))

for name, coef in zip(features, lin.coef_):
    print(name, round(coef, 3))

print("intercept", round(lin.intercept_, 3))

# Random Forest

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("forest train", mean_absolute_error(y_train, rf.predict(X_train)))
print("forest val", mean_absolute_error(y_val, rf.predict(X_val)))

print(X_train.describe().loc[["min", "max"]].round(2))
print(X_val.describe().loc[["min", "max"]].round(2))

print("train thrust range:", y_train.min().round(2), y_train.max().round(2))
print("forest predictions:", rf.predict(X_val).min().round(2), rf.predict(X_val).max().round(2))
print("true val thrust:", y_val.min().round(2), y_val.max().round(2))

# Decision Tree

depths = range(1, 21)
train_errors = []
val_errors = []

for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)

    train_errors.append(mean_absolute_error(y_train, tree.predict(X_train)))
    val_errors.append(mean_absolute_error(y_val, tree.predict(X_val)))

    print(d, round(train_errors[-1], 3), round(val_errors[-1], 3))

plt.figure()
plt.plot(list(depths), train_errors, marker='o', label="train")
plt.plot(list(depths), val_errors, marker='o', label="validation")
plt.axhline(y=0, color="green", linestyle="--", label="linear regression (val)")
plt.xlabel("max depth")
plt.ylabel("mean absolute error (N)")
plt.title("Decision Tree: train vs validation error by depth")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("overfitting_curve.png", dpi=150, bbox_inches="tight")

# Cross-validation with GroupKFold

dev =df[~df["flight"].isin(test_flights)] # keep every row that is not in the test flights (~ converts all TRUE values to FALSE and the opposite)

X_dev = dev[features]
y_dev = dev[label]

groups = dev["flight"] # grouping variable for GroupKFold
gkf = GroupKFold(n_splits=4) # 16 flights in dev, so 4 splits will give 4 flights per fold

rf_scores = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42), X_dev, y_dev, groups=groups, cv=gkf, scoring="neg_mean_absolute_error")

print("forest folds:", (-rf_scores).round(3))
print("forest mean:", (-rf_scores.mean()).round(3), "std:", rf_scores.std().round(3))

lin_scores = cross_val_score(LinearRegression(), X_dev, y_dev, groups=groups, cv=gkf, scoring="neg_mean_absolute_error")

print("linear folds:", (-lin_scores).round(6))
print("linear mean:", (-lin_scores.mean()).round(6), "std:", lin_scores.std().round(6))


# Classification: is the drone closing on its target?

df["closing"] = ((df["x"] - df["target_x"]) * df["velocity_x"] + (df["y"] - df["target_y"]) * df["velocity_y"]) < 0  # dot prodcuct of (position - target) with velocity

print(df["closing"].value_counts()) # number of values that are true (closing) and false (moving away)
print(df["closing"].mean().round(3))

y_train_c = df.loc[df["flight"].isin(train_flights), "closing"]
y_val_c = df.loc[df["flight"].isin(val_flights), "closing"]
y_test_c = df.loc[df["flight"].isin(test_flights), "closing"]

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train_c)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train_c)

print("baseline(always True):", round(y_val_c.mean(), 3))
print("logistic train:", round(accuracy_score(y_train_c, logreg.predict(X_train)), 3))
print("logistic val:", round(accuracy_score(y_val_c, logreg.predict(X_val)), 3))
print("forest train:", round(accuracy_score(y_train_c, rfc.predict(X_train)), 3))
print("forest val:", round(accuracy_score(y_val_c, rfc.predict(X_val)), 3))

print (confusion_matrix(y_val_c, logreg.predict(X_val)))
print (confusion_matrix(y_val_c, rfc.predict(X_val)))

speed = np.sqrt(df["velocity_x"]**2 + df["velocity_y"]**2)

print("slow rows:", df.loc[speed < 0.1, "closing"].mean().round(3))
print("fast rows:", df.loc[speed >= 0.1, "closing"].mean().round(3))

df["step"] = df.index % 500
slow = speed < 0.1

print("how many slow rows:", slow.sum())
print(df.loc[slow, "step"].describe().round(1))


# Final Test: Linear Regression for thrust_x and Random Forest for closing

y_dev_c = df.loc[~df["flight"].isin(test_flights), "closing"]

final_lin = LinearRegression()
final_lin.fit(X_dev, y_dev)

final_rfc = RandomForestClassifier(n_estimators=100, random_state=42)
final_rfc.fit(X_dev, y_dev_c)

print("Final Regression MAE(N):", mean_absolute_error(y_test, final_lin.predict(X_test)))
print("Final Classification Accuracy:", round(accuracy_score(y_test_c, final_rfc.predict(X_test)), 3))
print("test baseline:", round(y_test_c.mean(), 3))
print(confusion_matrix(y_test_c, final_rfc.predict(X_test)))