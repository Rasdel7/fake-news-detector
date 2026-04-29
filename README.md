# Fake News Detector 📰

NLP model that detects fake news articles with 98.5% accuracy,
trained on 44,898 real and fake news articles.

## Live Demo
[Click here](https://fake-news-detector-5t89zi8iykspvqbfaoxec9.streamlit.app)

## Features
- Paste any article and get instant real/fake prediction
- Confidence score with visual bar
- Test with example headlines
- Full explanation of how the model works

## Model Details
- Algorithm: Logistic Regression + TF-IDF
- Training data: 44,898 articles
- Accuracy: 98.5%
- Vocabulary: 50,000 features

## Tools Used
- Python, Scikit-learn, Streamlit, Pandas, Matplotlib

## How to Run Locally
pip install streamlit scikit-learn pandas numpy matplotlib
python3 train_model.py
streamlit run app.py
