import pickle

# Load the model and vectorizer
def load_model():
    try:
        # Load the trained model
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        # Load the vectorizer
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        raise Exception("Model or vectorizer file not found. Make sure they are in the project directory.")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

# Function to predict spam/ham
def predict_spam(email_content, model, vectorizer):
    if not email_content.strip():
        return "Empty input"
    
    input_data = vectorizer.transform([email_content])
    prediction = model.predict(input_data)
    return "Ham" if prediction[0] == 1 else "Spam"
