from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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

# Convert text data into numerical form using CountVectorizer
vectorizer_n = CountVectorizer()
X = vectorizer_n.fit_transform(emails)

# Save the vectorizer
joblib.dump(vectorizer_n, "count_vectorizer.pkl")  # Save vectorizer

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train Naïve Bayes model
model_n = MultinomialNB()
model_n.fit(X_train, y_train)
# Save the model
joblib.dump(model_n, "naive.pkl")


test_email = ["Congratulations! You have won a free iPhone. Click here to claim."]


test_vector = vectorizer_n.transform(test_email)
print("Naive Bayes Prediction:", model_n.predict(test_vector))
