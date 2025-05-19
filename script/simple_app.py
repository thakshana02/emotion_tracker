import os
import pickle
import re
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Create a simple emotion classification model for demonstration
class SimpleEmotionClassifier:
    def __init__(self):
        # Define emotion mapping
        self.emotion_mapping = {
            0: 'sadness',
            1: 'joy',
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        }
        
        # Create and fit a simple TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            min_df=1,
            sublinear_tf=True
        )
        
        # Sample training data (example sentences for each emotion)
        sample_texts = [
            # Sadness
            "I feel so sad and lonely today",
            "I miss my old friends so much",
            "I feel depressed and hopeless",
            "I'm feeling down and blue",
            "I'm heartbroken over the loss",
            
            # Joy
            "I'm so happy and excited about the news",
            "I feel wonderful and joyful today",
            "I'm delighted with the results",
            "I feel so accomplished and proud",
            "I'm thrilled to have achieved this goal",
            
            # Love
            "I love my family more than anything",
            "I feel so much affection for her",
            "I'm deeply in love with you",
            "I cherish our relationship",
            "I adore spending time with my children",
            
            # Anger
            "I'm furious about what happened",
            "I feel so angry and frustrated",
            "I'm irritated by their behavior",
            "I'm outraged by this decision",
            "I resent the way I was treated",
            
            # Fear
            "I'm terrified of what might happen",
            "I feel anxious about the future",
            "I'm scared of being alone",
            "I have a dreadful feeling about this",
            "I'm worried about the outcome",
            
            # Surprise
            "I was completely shocked by the news",
            "I'm amazed at the results",
            "I was stunned when I heard",
            "I'm astonished by what happened",
            "I can't believe how surprising this is"
        ]
        
        # Create labels (6 examples per emotion)
        sample_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        
        # Fit the vectorizer
        X = self.vectorizer.fit_transform(sample_texts)
        
        # Create and fit a simple logistic regression model
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )
        self.model.fit(X, sample_labels)
        
    def preprocess_text(self, text):
        """Preprocess text for prediction"""
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
        
        return text
    
    def predict(self, text):
        """Predict emotion from text"""
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Transform to TF-IDF features
        X = self.vectorizer.transform([processed_text])
        
        # Predict emotion
        emotion_id = self.model.predict(X)[0]
        emotion_name = self.emotion_mapping[emotion_id]
        
        # Get probability scores
        proba = self.model.predict_proba(X)[0]
        
        # Create dictionary of emotion probabilities
        emotion_probs = {self.emotion_mapping[i]: float(proba[idx]) for idx, i in enumerate(self.model.classes_)}
        
        return emotion_name, emotion_probs
        
# Create a simple hardcoded dictionary for demo purposes
# This will help classify common patterns reliably
emotion_keywords = {
    'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'heartbroken', 'crying', 'tears', 'grief', 'lonely', 'devastated', 'gloomy', 'upset'],
    'joy': ['happy', 'joyful', 'excited', 'delighted', 'thrilled', 'pleased', 'cheerful', 'glad', 'content', 'celebration', 'wonderful', 'great'],
    'love': ['love', 'affection', 'caring', 'adore', 'cherish', 'fondness', 'romantic', 'attachment', 'devotion', 'passion', 'relationship'],
    'anger': ['angry', 'furious', 'irritated', 'annoyed', 'enraged', 'mad', 'frustrated', 'outraged', 'bitter', 'resentful', 'hostile', 'offended'],
    'fear': ['afraid', 'scared', 'terrified', 'fearful', 'worried', 'anxious', 'nervous', 'frightened', 'panic', 'terror', 'dread', 'concerned'],
    'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'speechless', 'unexpected', 'startled', 'astounded', 'disbelief']
}

# Initialize the classifier
classifier = SimpleEmotionClassifier()

def keyword_enhanced_prediction(text, model_prediction, model_probabilities):
    """Enhance prediction with keyword matching for better reliability"""
    # Count keyword occurrences for each emotion
    keyword_counts = {emotion: 0 for emotion in emotion_keywords}
    
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', text.lower()):
                keyword_counts[emotion] += 1
    
    # If strong keyword evidence exists, boost that emotion's probability
    max_count = max(keyword_counts.values())
    if max_count > 1:  # If we have multiple matches for an emotion
        max_emotion = max(keyword_counts, key=keyword_counts.get)
        
        # If model's top prediction doesn't match the keyword-based prediction
        # and the keyword evidence is strong, return the keyword-based prediction
        if model_prediction != max_emotion and max_count >= 3:
            return max_emotion, {e: (0.9 if e == max_emotion else 0.02) for e in emotion_keywords}
    
    # Otherwise, return the model's prediction
    return model_prediction, model_probabilities

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the request
        text = request.form['text']
        
        # Get model prediction
        emotion, probabilities = classifier.predict(text)
        
        # Enhance with keyword matching for more reliability
        emotion, probabilities = keyword_enhanced_prediction(text, emotion, probabilities)
        
        # Sort probabilities
        sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'probabilities': sorted_probs
        })
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
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
    app.run(debug=True, port=5000)