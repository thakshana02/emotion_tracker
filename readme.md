Emotion Classification System
A machine learning-based tool for detecting emotions in text using natural language processing techniques.
Show Image
Overview
This project implements a text-based emotion classifier that can identify six distinct emotions from text input:

Sadness
Joy
Love
Anger
Fear
Surprise

The system uses a combination of advanced NLP techniques including TF-IDF vectorization with n-grams and logistic regression to accurately classify emotional content in text.
Model Performance
Our model achieves impressive accuracy across various emotion categories:
MetricScoreOverall Accuracy89.55%Macro F1-score0.86Weighted F1-score0.90
Performance by Emotion Class
Show Image
EmotionPrecisionRecallF1-ScoreSadness0.950.910.93Joy0.930.900.92Love0.720.890.80Anger0.890.900.89Fear0.920.830.87Surprise0.640.880.74
Feature Importance
Analysis of our model revealed the most important features for each emotion:
Sadness

melancholy (4.47)
punished (4.43)
lethargic (4.30)
exhausted (4.10)
miserable (4.04)

Joy

successful (3.98)
popular (3.88)
resolved (3.85)
innocent (3.83)
pleasant (3.82)

Love

caring (8.64)
sympathetic (8.21)
loving (8.13)
supportive (8.00)
longing (7.92)

Anger

dangerous (6.96)
fucked (6.60)
irritable (6.52)
bothered (6.51)
greedy (6.51)

Fear

shaken (7.65)
terrified (7.50)
reluctant (7.00)
vulnerable (6.79)
paranoid (6.61)

Surprise

impressed (12.44)
amazed (11.92)
surprised (11.44)
curious (11.33)
shocked (10.60)

Web Application
The system is implemented as a web application using Flask, providing an intuitive interface for emotion classification.
Show Image
How to Use

Enter or paste text in the input field
Click "Analyze Emotion"
View the predicted emotion and confidence scores

Technical Details

Framework: Flask
ML Library: scikit-learn
Feature Extraction: TF-IDF with n-grams (1-3)
Model: Logistic Regression with class balancing
Text Preprocessing: Lowercase conversion, punctuation removal, stopword filtering

Installation and Setup

Clone the repository:

git clone https://github.com/thakshana02/emotion_tracker.git    
cd emotion-tracker

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

Open your browser and navigate to:

http://127.0.0.1:5000/
Dataset
The model was trained on a large dataset of text samples labeled with emotions. The dataset includes:

Training set: ~33,000 samples
Validation set: ~5,000 samples
Test set: ~2,000 samples

Applications
This emotion classification system can be used for various applications:

Social Media Analytics: Track emotional responses to brands, products, or campaigns
Customer Experience: Prioritize customer service based on emotional urgency
Content Creation: Analyze which content evokes desired emotional responses
Mental Health Applications: Help track emotional patterns in text over time
Education & Research: Support studies on emotional expression in text

Future Improvements

Implement multi-label classification for text with mixed emotions
Add emotion intensity measurement
Expand the model to support more languages
Improve preprocessing for social media specific text