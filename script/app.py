import os
import pickle
import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Define paths
model_dir = r"D:\Projects\emotion_tracker\dataset\saved_model"

# Define emotion mapping
emotion_mapping = {
    0: 'sadness',
    1: 'joy', 
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Try to load the trained model
try:
    print("Loading classifier model...")
    with open(os.path.join(model_dir, 'emotion_classifier_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Create a new TF-IDF vectorizer since the saved one is causing problems
print("Creating new vectorizer with similar parameters to the trained one...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=10000,
    min_df=5,
    max_df=0.8,
    sublinear_tf=True
)

# Load some sample texts to fit the vectorizer
sample_texts = [
    # Sadness
    "I feel so sad and depressed today",
    "I miss my old friends so much",
    "I feel heartbroken and abandoned",
    "I'm so lonely and miserable",
    "Everything feels gloomy and hopeless",
    
    # Joy
    "I'm so happy and excited today",
    "I feel wonderful and joyful",
    "I'm thrilled about the good news",
    "I feel so accomplished and proud",
    "Today was absolutely amazing",
    
    # Love
    "I love my family so much",
    "I feel such affection for you",
    "I'm deeply in love with her",
    "The feeling of being loved is incredible",
    "I adore spending time with you",
    
    # Anger
    "I'm so angry about what happened",
    "I feel furious and frustrated",
    "I'm irritated by their behavior",
    "This makes me so mad",
    "I'm outraged by this decision",
    
    # Fear
    "I'm terrified of what might happen",
    "I feel so anxious and scared",
    "I'm afraid of being alone",
    "This situation makes me nervous",
    "I'm worried about the future",
    
    # Surprise
    "I was completely shocked by the news",
    "I'm amazed at what happened",
    "I can't believe how surprising this is",
    "That was totally unexpected",
    "I was stunned when I heard"
]

# Fit the vectorizer
vectorizer.fit(sample_texts)
print("Vectorizer fitted successfully")

# Define preprocessing function
def preprocess_text(text):
    """Simplified preprocessing function"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle common abbreviations
    abbreviations = {
        'u': 'you',
        'r': 'are',
        'ur': 'your',
        'n': 'and',
        'y': 'why',
        'im': 'i am',
        'ive': 'i have',
        'ill': 'i will',
        'dont': 'do not',
        'doesnt': 'does not',
        'didnt': 'did not',
        'cant': 'can not',
        'wont': 'will not',
        'amp': 'and',
    }
    
    for abbr, full in abbreviations.items():
        text = re.sub(r'\b' + abbr + r'\b', full, text)
    
    return text

# Keywords for emotion detection as a fallback
emotion_keywords = {
    'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'heartbroken', 'crying', 'tears', 'grief', 'lonely', 'devastated', 'gloomy', 'upset', 'sorrow', 'pain', 'broken', 'hurt', 'loss', 'tragedy', 'grief', 'pain', 'suffer'],
    'joy': ['happy', 'joyful', 'excited', 'delighted', 'thrilled', 'pleased', 'cheerful', 'glad', 'content', 'celebration', 'wonderful', 'great', 'smile', 'laugh', 'joy', 'celebrate', 'happiness'],
    'love': ['love', 'affection', 'caring', 'adore', 'cherish', 'fondness', 'romantic', 'attachment', 'devotion', 'passion', 'relationship', 'embrace', 'warmth', 'tenderness', 'heart'],
    'anger': ['angry', 'furious', 'irritated', 'annoyed', 'enraged', 'mad', 'frustrated', 'outraged', 'bitter', 'resentful', 'hostile', 'offended', 'rage', 'fury', 'hate', 'hatred', 'violent', 'upset'],
    'fear': ['afraid', 'scared', 'terrified', 'fearful', 'worried', 'anxious', 'nervous', 'frightened', 'panic', 'terror', 'dread', 'concerned', 'horror', 'threat', 'worry', 'paranoia', 'panic'],
    'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'speechless', 'unexpected', 'startled', 'astounded', 'disbelief', 'wonder', 'shock', 'unexpected', 'unbelievable']
}

# Define prediction function that leverages both the trained model (if available) and keyword matching
def predict_emotion(text):
    """Predict emotion from text"""
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform to TF-IDF features
    X = vectorizer.transform([processed_text])
    
    # Get keyword-based scores
    keyword_scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = 0
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', processed_text):
                score += 1
        keyword_scores[emotion] = score
    
    # Normalize keyword scores
    total_score = sum(keyword_scores.values()) or 1
    for emotion in keyword_scores:
        keyword_scores[emotion] = keyword_scores[emotion] / total_score
    
    if model is not None:
        try:
            # Try to use the trained model
            emotion_id = model.predict(X)[0]
            emotion_name = emotion_mapping[emotion_id]
            
            # Get probability scores
            proba = model.predict_proba(X)[0]
            
            # Create dictionary of emotion probabilities
            emotion_probs = {emotion_mapping[i]: float(proba[idx]) for idx, i in enumerate(model.classes_)}
            
            # Combine model probabilities with keyword scores (70% model, 30% keywords)
            combined_probs = {}
            for emotion in emotion_probs:
                combined_probs[emotion] = 0.7 * emotion_probs[emotion] + 0.3 * keyword_scores.get(emotion, 0)
            
            # Normalize combined probabilities
            total = sum(combined_probs.values())
            for emotion in combined_probs:
                combined_probs[emotion] /= total
            
            # Get top emotion
            emotion_name = max(combined_probs, key=combined_probs.get)
            
            return emotion_name, combined_probs
            
        except Exception as e:
            print(f"Error using trained model: {str(e)}")
            # Fall back to keyword-based prediction
            
    # Use keyword-based prediction
    emotion_name = max(keyword_scores, key=keyword_scores.get)
    return emotion_name, keyword_scores

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the request
        text = request.form['text']
        
        # Make prediction
        emotion, probabilities = predict_emotion(text)
        
        # Sort probabilities
        sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'probabilities': sorted_probs
        })
    except Exception as e:
        print(f"Error in prediction endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Create templates folder in the correct directory
    script_dir = r"D:\Projects\emotion_tracker\script"
    templates_dir = os.path.join(script_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html if it doesn't exist
    index_path = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_path):
        with open(index_path, 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #336699;
            text-align: center;
        }
        .container {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 10px;
            box-sizing: border-box;
            border: 1px solid #ddd;
            border-radius: 5px;
            height: 150px;
            margin-top: 10px;
            margin-bottom: 20px;
            font-family: Arial, sans-serif;
        }
        button {
            background-color: #336699;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #254e77;
        }
        .result-container {
            display: none;
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .emotion-result {
            font-size: 24px;
            font-weight: bold;
            color: #336699;
            text-align: center;
            text-transform: uppercase;
            margin-bottom: 20px;
        }
        .progress-container {
            margin-bottom: 10px;
        }
        .emotion-label {
            display: inline-block;
            width: 80px;
            text-align: right;
            margin-right: 10px;
            font-weight: bold;
        }
        .progress-bar {
            height: 25px;
            background-color: #e9ecef;
            border-radius: 5px;
            position: relative;
            width: calc(100% - 100px);
            display: inline-block;
            vertical-align: middle;
        }
        .progress-fill {
            height: 100%;
            border-radius: 5px;
            background-color: #336699;
            text-align: right;
            color: white;
            line-height: 25px;
            padding-right: 10px;
            box-sizing: border-box;
        }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #336699;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error-message {
            color: #e63946;
            background-color: #f8d7da;
            border: 1px solid #f5c2c7;
            border-radius: 5px;
            padding: 10px;
            margin-top: 20px;
            display: none;
        }
    </style>
</head>
<body>
    <h1>Emotion Classification Tool</h1>
    
    <div class="container">
        <h2>Enter text to analyze:</h2>
        <textarea id="textInput" placeholder="Type or paste a paragraph to classify the emotion..."></textarea>
        <button id="analyzeBtn">Analyze Emotion</button>
        <div class="loader" id="loader"></div>
        <div class="error-message" id="errorMessage"></div>
    </div>
    
    <div class="result-container" id="resultContainer">
        <div class="emotion-result" id="emotionResult"></div>
        <h3>Confidence Scores:</h3>
        <div id="probabilitiesResult"></div>
    </div>
    
    <script>
        document.getElementById('analyzeBtn').addEventListener('click', function() {
            const text = document.getElementById('textInput').value.trim();
            
            if (!text) {
                alert('Please enter some text to analyze.');
                return;
            }
            
            // Show loader
            document.getElementById('loader').style.display = 'block';
            
            // Hide results and error
            document.getElementById('resultContainer').style.display = 'none';
            document.getElementById('errorMessage').style.display = 'none';
            
            // Send the text to the server for analysis
            const formData = new FormData();
            formData.append('text', text);
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loader
                document.getElementById('loader').style.display = 'none';
                
                if (data.success) {
                    // Display the results
                    document.getElementById('emotionResult').textContent = data.emotion;
                    
                    // Display probability bars
                    const probsDiv = document.getElementById('probabilitiesResult');
                    probsDiv.innerHTML = '';
                    
                    Object.entries(data.probabilities).forEach(([emotion, probability]) => {
                        const percentage = Math.round(probability * 100);
                        const color = emotion === data.emotion ? '#336699' : '#adb5bd';
                        
                        const container = document.createElement('div');
                        container.className = 'progress-container';
                        
                        const label = document.createElement('span');
                        label.className = 'emotion-label';
                        label.textContent = emotion + ':';
                        
                        const progressBar = document.createElement('div');
                        progressBar.className = 'progress-bar';
                        
                        const progressFill = document.createElement('div');
                        progressFill.className = 'progress-fill';
                        progressFill.style.width = percentage + '%';
                        progressFill.style.backgroundColor = color;
                        progressFill.textContent = percentage + '%';
                        
                        progressBar.appendChild(progressFill);
                        container.appendChild(label);
                        container.appendChild(progressBar);
                        
                        probsDiv.appendChild(container);
                    });
                    
                    // Show result container
                    document.getElementById('resultContainer').style.display = 'block';
                } else {
                    // Display error
                    const errorDiv = document.getElementById('errorMessage');
                    errorDiv.textContent = 'Error: ' + data.error;
                    errorDiv.style.display = 'block';
                }
            })
            .catch(error => {
                // Hide loader
                document.getElementById('loader').style.display = 'none';
                
                // Display error
                const errorDiv = document.getElementById('errorMessage');
                errorDiv.textContent = 'An error occurred: ' + error;
                errorDiv.style.display = 'block';
            });
        });
    </script>
</body>
</html>''')
    
    print("Starting Flask app...")
    # Start the Flask app
    app.run(debug=True, port=5000)