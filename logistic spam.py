from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder




# Load data
data = pd.read_excel("spam_emails.xlsx")
df = pd.DataFrame(data)
emails = df["text"]
labels = df["label"]
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
# Convert text data into numerical form using TfidfVectorizer
vectorizer_p = TfidfVectorizer()
X = vectorizer_p.fit_transform(emails)

# Save the vectorizer
joblib.dump(vectorizer_p, "tfidf_vectorizer.pkl")  # Save vectorizer

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train Logistic Regression model
model_p = LogisticRegression()
model_p.fit(X_train, y_train)

# Save the model
joblib.dump(model_p, "regression.pkl")



