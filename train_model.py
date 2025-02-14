import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from test import TextToNum  # Import the TextToNum class

# Load the dataset
data = pd.read_csv('file.csv')

# Print the column names to verify
print(data.columns)

# Preprocess the text data
def preprocess_text(text):
    text_processor = TextToNum(text)
    text_processor.cleaner()
    text_processor.token()
    text_processor.removeStop()
    processed_text = text_processor.stemme()
    return " ".join(processed_text)

# Apply preprocessing to the text data
data['processed_text'] = data['tweets'].apply(preprocess_text)

# Extract the processed text and labels
texts = data['processed_text'].values
labels = data['labels'].values

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a pipeline with a TfidfVectorizer and a LogisticRegression classifier
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model
pipeline.fit(train_texts, train_labels)

# Evaluate the model
accuracy = pipeline.score(test_texts, test_labels)
print(f"Model accuracy: {accuracy:.2f}")

# Save the model and vectorizer to disk
with open('model.pickle', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

print("Model trained and saved to model.pickle")