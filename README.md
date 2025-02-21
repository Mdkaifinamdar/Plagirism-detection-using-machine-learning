# PLAGIRISM-DETECTION-USING-MACHINE-LEARNING
This project implements a plagiarism detection system utilizing advanced machine learning and natural language processing techniques to detect and address complex plagiarism types. It combines traditional string matching with deep learning and cosine similarity to identify paraphrasing, synonym replacement, sentence reordering, and even cross-language plagiarism.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Model Details](#model-details)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Introduction

In today's digital world, plagiarism detection has become essential across academia, journalism, and other content-driven fields. Traditional plagiarism detection methods fail to detect sophisticated plagiarism techniques like paraphrasing or cross-language copying. This project leverages machine learning models, such as NLP-based deep learning and cosine similarity, to overcome these challenges by understanding text semantics and context.

## Features

- **Paraphrasing Detection**: Identifies content that has been reworded or uses synonyms.
- **Cross-Language Plagiarism Detection**: Detects plagiarism even if content is translated into another language.
- **Sentence Reordering**: Recognizes content even when the sentence order is altered.
- **Cosine Similarity for Similarity Matching**: Uses cosine similarity to compare text vectors, improving detection accuracy.
- **Scalability**: Can handle large datasets efficiently through cloud support and distributed processing.

## Project Structure

```
.
├── app.py                                  # Main application script
├── Building Plagiarism checker using Machine Learning.ipynb   # Jupyter notebook with project code and analysis
├── dataset.csv                             # Dataset used for training and testing
├── model.pkl                               # Saved ML model for reuse
├── tfidf_vectorizer.pkl                    # Saved TF-IDF vectorizer for text processing
├── templates                               # Directory for HTML templates
│   └── index.html                          # HTML file for web interface
└── README.md                               # Project readme file
```

## Setup and Installation

Clone the repository:
git clone https://github.com/Mdkaifinamdar/PLAGIRISM-DETECTION-USING-MACHINE-LEARNING.git
cd PLAGIRISM-DETECTION-USING-MACHINE-LEARNING

Install Flask:
pip install Flask

Run the Jupyter Notebook:
Open "Building Plagiarism checker using Machine Learning.ipynb" in Jupyter Notebook and execute all cells to load the model and vectorizer.
![image](https://github.com/user-attachments/assets/01d5d538-4590-4d1b-8339-5370bf7f4ad4)


Run the Application:
Execute "app.py" using any code IDE like VS code or using terminal

Access the Application: After running app.py, the IP address and port displayed in the terminal. Open it in a web browser to access the application on your local machine.
![image](https://github.com/user-attachments/assets/858d44c1-4ffb-4b80-951d-cb88487ccfc4)
![image](https://github.com/user-attachments/assets/a9f07531-8880-4b43-a88b-59aab2c195c8)


## Model Details

This project employs various machine learning models and techniques, including:
- **TF-IDF Vectorization** to convert text data into numerical representations.
- **Cosine Similarity** for calculating the similarity score between text vectors, enhancing the ability to detect paraphrased content.
- **Support Vector Machines (SVM)** for binary classification of text as plagiarized or original.

The models are evaluated based on accuracy, precision, recall, and F1-score to ensure high-quality detection of nuanced plagiarism.

## Results

Our evaluation shows significant improvement in detecting plagiarism compared to traditional methods:
- **Accuracy**: >90%
- **Precision**: 85-90%
- **Recall**: 88-92%
- **F1-score**: 89%

See `Building Plagiarism checker using Machine Learning.ipynb` for detailed analysis and performance metrics.

## Future Improvements

- **Extended Cross-Language Detection**: Expand support to more languages using multilingual transformers.
- **AI-Generated Content Detection**: Identify and differentiate human-written vs. AI-generated content.
- **Multimedia Plagiarism**: Incorporate multimedia content, like images and audio, to detect plagiarism beyond text.
