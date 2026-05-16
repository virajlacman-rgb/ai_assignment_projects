import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load data
df = pd.read_csv("customer_churn_nn.csv")

# 2. Basic exploration
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df["churn"].value_counts())
print(df["churn"].value_counts(normalize=True))

# 3. Remove customer ID
df = df.drop("customer_id", axis=1)

# 4. Convert categorical columns into numbers
df = pd.get_dummies(df, drop_first=True)

# 5. Features and target
X = df.drop("churn", axis=1)
y = df["churn"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Handle imbalance using SMOTE
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(y_train.value_counts())

# 8. Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Build neural network
model = Sequential()

model.add(Dense(16, activation="relu", input_dim=X_train.shape[1]))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

# 10. Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 11. Train model
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
)

# 12. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# 13. Predictions
y_pred = model.predict(X_test)

y_pred_class = (y_pred > 0.5).astype(int)

# 14. Confusion matrix and report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))

print("Classification Report:")
print(classification_report(y_test, y_pred_class))
print(df.corr()["churn"].sort_values(ascending=False))