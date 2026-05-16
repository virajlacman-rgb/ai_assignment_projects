import pandas as pd

# load dataset
df = pd.read_csv("customer_support_text_classification.csv")

# show column names first
#print("Column Names:")
#print(df.columns)

# show first 5 rows
#print("\nFirst 5 Rows:")
#print(df.head())
# dataset shape
#print("\nDataset Shape:")
#print(df.shape)

# missing values
#print("\nMissing Values:")
#print(df.isnull().sum())

# sentiment distribution
#print("\nSentiment Distribution:")
#print(df["sentiment_label"].value_counts())
import re

# text cleaning function
def clean_text(text):

    # lowercase
    text = text.lower()

    # remove numbers and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text

# apply cleaning
df["clean_message"] = df["customer_message"].apply(clean_text)

# show original vs cleaned
#print("\nOriginal Message:")
#print(df["customer_message"].iloc[0])

#print("\nCleaned Message:")
#print(df["clean_message"].iloc[0])
from sklearn.feature_extraction.text import TfidfVectorizer

# create vectorizer
vectorizer = TfidfVectorizer()

# convert text into numbers
X = vectorizer.fit_transform(df["clean_message"])

# target column
y = df["sentiment_label"]

# shape of vectorized data
#print("\nVectorized Shape:")
#print(X.shape)
from sklearn.model_selection import train_test_split

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# shapes
#print("\nTraining Shape:")
#print(X_train.shape)

#print("\nTesting Shape:")
#print(X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# create model
model = LogisticRegression()

# train model
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)

#print("\nModel Accuracy:")
#print(accuracy)
from sklearn.metrics import classification_report

#print("\nClassification Report:")
#print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

#print("\nConfusion Matrix:")
#print(cm)
# sample custom messages
sample_messages = [
    "I am very happy with your service",
    "My order has not arrived yet",
    "Please help me reset my password"
]

# clean messages
clean_samples = [clean_text(msg) for msg in sample_messages]

# convert into TF-IDF
sample_vectors = vectorizer.transform(clean_samples)

# predictions
predictions = model.predict(sample_vectors)

# show results
print("\nSample Predictions:")

for msg, pred in zip(sample_messages, predictions):
    print(f"Message: {msg}")
    print(f"Predicted Sentiment: {pred}")
    print("-" * 40)