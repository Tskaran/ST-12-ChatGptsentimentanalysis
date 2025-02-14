from flask import Flask, render_template, request
import pickle
import os
from test import TextToNum  # Import the TextToNum class

app = Flask(__name__)

# Define the path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'model.pickle')

# Load the model with error handling
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    model = None
    print("Model file not found. Please ensure 'model.pickle' exists.")
except Exception as e:
    model = None
    print(f"An error occurred while loading the model: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        user_input = request.form["message"]
        
        # Preprocess the user input
        text_processor = TextToNum(user_input)
        text_processor.cleaner()
        text_processor.token()
        text_processor.removeStop()
        processed_text = text_processor.stemme()
        
        if model is None:
            return render_template("result.html", sentiment="Error: Model not loaded.", user_input=user_input)

        # Vectorize the input
        try:
            input_vector = model.named_steps['tfidfvectorizer'].transform([" ".join(processed_text)])
            # Make prediction
            prediction = model.named_steps['logisticregression'].predict(input_vector)
        except Exception as e:
            return render_template("result.html", sentiment="Error during prediction: " + str(e), user_input=user_input)
        sentiment = "Neutral"
        if prediction[0] == 'good':
            sentiment = "Positive"
        elif prediction[0] == 'bad':
            sentiment = "Negative"
        elif prediction[0] == 'neutral':
            sentiment = "Neutral"
        
        return render_template("result.html", sentiment=sentiment, user_input=user_input)
    elif request.method == "GET":
        return render_template("predict.html", sentiment="No input provided", user_input="")

if __name__ == "__main__":
    app.run(debug=True)