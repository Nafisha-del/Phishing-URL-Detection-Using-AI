from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# Initialize the app
app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer
model_path = 'C:/Users/Windows 11/python_codes/BERT model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# Load the Random Forest model
rf_model_path = 'C:/Users/Windows 11/python_codes/RF model/random_forest_model_with_smote.pkl'
vectorizer_path = 'C:/Users/Windows 11/python_codes/RF model/vectorizer.pkl'
rf_model = joblib.load(rf_model_path)
vectorizer = joblib.load(vectorizer_path)

# Function to predict phishing or not
def predict_phishing(url):
    inputs = tokenizer(text=url, return_tensors='pt', padding=True, truncation=True)
    print("--------INPUTS-----------", inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        predicted_class = torch.argmax(logits, dim=1).item()
        print("--------PREDICTED CLASS-----------", predicted_class)
        print("--------PREDICTED Prob-----------", probabilities)

        result = {
            'prediction': "Phishing URL" if predicted_class == 0 else "Safe URL",
            'probabilities': {
                'safe': probabilities[1],
                'phishing': probabilities[0]
            }
        }
    return result

# Feature extraction function 
def extract_features_from_url(url):
    features = vectorizer.transform([url]).toarray()[0]
    print(f"TF-IDF Features: {features}")
    return features


# Function to predict phishing or not
def rf_predict_phishing(url):
    features = extract_features_from_url(url)
    prediction = rf_model.predict([features])
    probabilities = rf_model.predict_proba([features])[0]

    print(f"Features: {features}")
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {probabilities}")

    rf_result = {
        'prediction': "Phishing URL" if prediction == 0 else "Safe URL",
        'probabilities': {
            'safe': probabilities[1],
            'phishing': probabilities[0]
        }
    }
    return rf_result

# Flask route for the form submission
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Parse the incoming JSON request
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        print("--------URL-----------", url)
        # Predict phishing or not
        bert_result = predict_phishing(url)
        print("--------BERT RESULT-----------", bert_result)
        rf_result = rf_predict_phishing(url)
        print("--------RANDOM FOREST RESULT-----------", rf_result)
        combined_result = {
            'BERT_result': bert_result,
            'Random_Forest_result': rf_result
        }
        # return jsonify({'result': result})
        return jsonify(combined_result)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
