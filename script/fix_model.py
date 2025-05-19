import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Path to the saved model and vectorizer
model_dir = r"D:\Projects\emotion_tracker\dataset\saved_model"
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

# Load the vectorizer
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Check if the vectorizer is fitted
print("Checking vectorizer status:")
print(f"Vocabulary size: {len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else 'Not fitted'}")
print(f"IDF vector length: {len(vectorizer.idf_) if hasattr(vectorizer, 'idf_') else 'Not fitted'}")

# If the vectorizer is not properly fitted, we need a temporary solution
# This creates a simple vocabulary and IDF vector for testing
if not hasattr(vectorizer, 'idf_') or not hasattr(vectorizer, 'vocabulary_'):
    print("Fixing vectorizer...")
    
    # Sample documents to create a basic vocabulary
    sample_docs = [
        "I feel happy and joyful today",
        "I am feeling sad and depressed",
        "I am so angry right now",
        "I'm scared and terrified",
        "I feel surprised by this",
        "I love you so much"
    ]
    
    # Fit a new vectorizer on sample documents
    temp_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        min_df=1,
        max_df=0.8,
        sublinear_tf=True
    )
    temp_vectorizer.fit(sample_docs)
    
    # Copy the attributes to the original vectorizer
    if not hasattr(vectorizer, 'vocabulary_'):
        vectorizer.vocabulary_ = temp_vectorizer.vocabulary_
    
    if not hasattr(vectorizer, 'idf_'):
        vectorizer.idf_ = temp_vectorizer.idf_
    
    # Save the fixed vectorizer
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Vectorizer fixed and saved!")

print("Done!")