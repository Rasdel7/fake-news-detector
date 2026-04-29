import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Fake News Detector")
st.markdown("Paste any news article and find out if it's "
            "real or fake using Machine Learning.")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

try:
    model, tfidf = load_model()
    st.success("✅ Model loaded — 98.5% accuracy on 44,898 articles")
except:
    st.error("Model not found. Run train_model.py first.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Detect", "📊 Examples", "ℹ️ How It Works"
])

# Tab 1 — Detection
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Paste Your Article")
        title = st.text_input(
            "Article Title:",
            placeholder="Enter the headline here..."
        )
        text = st.text_area(
            "Article Text:",
            placeholder="Paste the full article text here...",
            height=250
        )

        if st.button("🔍 Analyze Article", type="primary"):
            if title.strip() or text.strip():
                content = (title + ' ' + text).lower().strip()
                vectorized  = tfidf.transform([content])
                prediction  = model.predict(vectorized)[0]
                probability = model.predict_proba(vectorized)[0]

                fake_prob = probability[0] * 100
                real_prob = probability[1] * 100

                st.markdown("---")
                if prediction == 1:
                    st.markdown(
                        "<h2 style='color:#2ecc71; "
                        "text-align:center'>✅ REAL NEWS</h2>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<h2 style='color:#e74c3c; "
                        "text-align:center'>🚨 FAKE NEWS</h2>",
                        unsafe_allow_html=True
                    )

                # Confidence bars
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh(['Real', 'Fake'],
                        [real_prob, fake_prob],
                        color=['#2ecc71', '#e74c3c'])
                for i, val in enumerate([real_prob, fake_prob]):
                    ax.text(val + 0.5, i, f'{val:.1f}%',
                            va='center', fontsize=11,
                            fontweight='bold')
                ax.set_xlim(0, 115)
                ax.set_title('Confidence Score', fontsize=12)
                ax.set_xlabel('Probability (%)')
                plt.tight_layout()
                st.pyplot(fig)

                # Warning
                if max(fake_prob, real_prob) < 70:
                    st.warning("⚠️ Low confidence — "
                               "verify with trusted sources.")
            else:
                st.warning("Please enter a title or text.")

    with col2:
        st.markdown("### 📌 Tips for Best Results")
        st.info("""
        **For accurate results:**
        - Paste the full article text
        - Include the headline
        - Longer text = more accurate

        **Always verify with:**
        - factcheck.org
        - snopes.com
        - Reuters fact check
        - AFP Fact Check
        """)

        st.markdown("### ⚠️ Disclaimer")
        st.warning("""
        This tool uses ML and is not 100% accurate.
        Always verify news from multiple
        trusted sources before sharing.
        """)

# Tab 2 — Examples
with tab2:
    st.markdown("### Test With Example Headlines")

    examples = {
        "Likely Real": [
            "Federal Reserve raises interest rates by 0.25 percent",
            "Scientists discover new treatment for Alzheimer's disease",
            "India GDP grows 7.2 percent in third quarter"
        ],
        "Likely Fake": [
            "BREAKING: Celebrity confirms they are secretly an alien",
            "Government puts secret mind control chemicals in water supply",
            "Doctors HATE this one weird trick that cures all diseases"
        ]
    }

    for category, headlines in examples.items():
        st.markdown(f"#### {category}")
        for headline in headlines:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {headline}")
            with col2:
                if st.button("Test", key=headline[:20]):
                    vec  = tfidf.transform([headline.lower()])
                    pred = model.predict(vec)[0]
                    prob = model.predict_proba(vec)[0]
                    if pred == 1:
                        st.success(f"Real ({prob[1]*100:.0f}%)")
                    else:
                        st.error(f"Fake ({prob[0]*100:.0f}%)")

# Tab 3 — How It Works
with tab3:
    st.markdown("### How This Works")

    st.markdown("""
    **Step 1 — Data**
    Trained on 44,898 news articles — 50% fake, 50% real.
    Dataset from Kaggle's Fake and Real News dataset.

    **Step 2 — Text Processing**
    Articles are cleaned, lowercased and combined
    (title + body text).

    **Step 3 — TF-IDF Vectorization**
    Converts text into numbers. TF-IDF gives higher weight
    to words that appear frequently in one article but
    rarely across all articles.

    **Step 4 — Logistic Regression**
    Trained on 35,918 articles, tested on 8,980 articles.
    Achieves 98.5% accuracy.

    **Step 5 — Prediction**
    New article is vectorized and classified as
    Real (1) or Fake (0) with a confidence score.
    """)

    # Model stats
    stats = pd.DataFrame({
        'Metric':   ['Training articles', 'Test articles',
                     'Accuracy', 'Vocabulary size', 'Algorithm'],
        'Value':    ['35,918', '8,980',
                     '98.5%', '50,000 features',
                     'Logistic Regression + TF-IDF']
    })
    st.dataframe(stats, use_container_width=True,
                 hide_index=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Model trained on 44,898 news articles | "
    "For educational purposes only"
)