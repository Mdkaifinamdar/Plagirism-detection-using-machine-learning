from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load and preprocess the dataset
data = pd.read_csv('dataset.csv')  
source_texts = data['source_text']  

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Preprocess and vectorize the source texts
source_vectors = tfidf_vectorizer.transform(source_texts)

def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    similarity_scores = cosine_similarity(vectorized_text, source_vectors)
    max_similarity = np.max(similarity_scores) * 100
    most_similar_index = np.argmax(similarity_scores)
    most_similar_text = source_texts.iloc[most_similar_index]
    
    threshold = 50
    result = "Plagiarism Detected" if max_similarity >= threshold else "No Plagiarism Detected"
    
    return result, max_similarity, most_similar_text if max_similarity >= threshold else None

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    detection_result, similarity_percentage, similar_text = detect(input_text)
    
    return render_template(
        'index.html', 
        result=f"{detection_result} ({similarity_percentage:.2f}% similarity)", 
        similar_text=similar_text
    )

if __name__ == "__main__":
    app.run(debug=True)
