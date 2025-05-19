Emotion Classifier
A machine learning system that detects emotions in text using NLP and logistic regression.
Show Image
🌟 Overview
This project implements a text-based emotion classifier that identifies six primary emotions:

😢 Sadness
😊 Joy
❤️ Love
😡 Anger
😨 Fear
😲 Surprise

The system achieves 89.55% accuracy using TF-IDF vectorization with n-grams and logistic regression.
📊 Model Performance
EmotionPrecisionRecallF1-ScoreSadness0.950.910.93Joy0.930.900.92Love0.720.890.80Anger0.890.900.89Fear0.920.830.87Surprise0.640.880.74
📂 Project Structure
emotion_tracker/
│
├── dataset/              # Data files
│   ├── training.csv
│   ├── validation.csv
│   ├── test.csv
│   └── saved_model/
│       ├── emotion_classifier_model.pkl
│       ├── tfidf_vectorizer.pkl
│       └── emotion_mapping.json
│
├── script/               # Application files
│   ├── app.py            # Flask application
│   ├── requirements.txt
│   └── templates/
│       └── index.html
│
├── screenshots/          # Application screenshots
│
└── README.md
💻 Installation

Clone the repository:

bashgit clone https://github.com/yourusername/emotion-classifier.git
cd emotion-classifier

Install dependencies:

bashcd script
pip install -r requirements.txt

Run the application:

bashpython app.py

Open your browser to:

http://127.0.0.1:5000/
🚀 How to Use

Enter text in the input field
Click "Analyze Emotion"
View the predicted emotion and confidence scores

Show Image
🔍 Key Features

Multi-class Emotion Detection: Accurately identifies 6 different emotions
TF-IDF with N-grams: Captures important word combinations and context
Class Balancing: Handles imbalanced datasets effectively
Detailed Confidence Scores: Provides probability for each emotion
Web Interface: Easy-to-use Flask application

🔧 Technical Details

Framework: Flask
ML Library: scikit-learn
Feature Extraction: TF-IDF with n-grams (1-3)
Model: Logistic Regression with class balancing
Text Preprocessing: Lowercase conversion, punctuation removal, stopword filtering

📈 Feature Importance
The model identified key words strongly associated with each emotion:
EmotionTop FeaturesSadnessmelancholy, punished, lethargic, exhausted, miserableJoysuccessful, popular, resolved, innocent, pleasantLovecaring, sympathetic, loving, supportive, longingAngerdangerous, fucked, irritable, bothered, greedyFearshaken, terrified, reluctant, vulnerable, paranoidSurpriseimpressed, amazed, surprised, curious, shocked
🔮 Applications

Social Media Analytics: Track emotional responses to brands and campaigns
Customer Service: Prioritize responses based on detected emotions
Content Creation: Analyze emotional impact of marketing materials
Mental Health: Support mood tracking and emotional awareness tools
User Experience: Evaluate emotional responses to products and services

🛣️ Future Improvements

Multi-label classification for mixed emotions
Emotion intensity measurement
Support for additional languages
Enhanced preprocessing for social media text
API for easier integration with other applications

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

CARER: Contextualized Affect Representations for Emotion Recognition - Research paper that influenced this work
The project was built with scikit-learn and Flask