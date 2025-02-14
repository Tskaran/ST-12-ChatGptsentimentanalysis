import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample data for sentiment analysis
data = {
    'text': [
        'I love ChatGPT',
        'ChatGPT is amazing',
        'I hate this',
        'This is terrible',
        'I am neutral about this'
    ],
    'sentiment': [1, 1, -1, -1, 0]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the data into features and labels
X = df['text']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Save the model and vectorizer to files
with open('ST-12-ChatGptsentimentanalysis/model.pickle', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('ST-12-ChatGptsentimentanalysis/vectorizer.pickle', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer have been created successfully.")
