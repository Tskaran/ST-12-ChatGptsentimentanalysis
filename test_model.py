import pickle
import os

# Define the path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'model.pickle')

# Load the model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Test the model with some sample inputs
sample_texts = ["I love this product!", "This is the worst experience ever."]
for text in sample_texts:
    input_vector = model.named_steps['tfidfvectorizer'].transform([text])
    prediction = model.named_steps['logisticregression'].predict(input_vector)
    print(f"Text: {text} -> Prediction: {prediction[0]}")