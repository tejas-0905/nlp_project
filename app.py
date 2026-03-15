
import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
.main {
background-color:#0f172a;
}
.title {
color:#38bdf8;
font-size:40px;
text-align:center;
font-weight:bold;
}
.subtitle{
text-align:center;
color:#cbd5f5;
font-size:18px;
}
.result-box{
padding:20px;
border-radius:10px;
text-align:center;
font-size:24px;
font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("sentiment_model.pkl","rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl","rb"))

# Header
st.markdown(
    '<p class="title" style="font-size:40px; font-weight:bold;">🎬 Movie Review Sentiment Analyzer</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle" style="font-size:20px;">AI powered NLP system to detect positive or negative movie reviews</p>',
    unsafe_allow_html=True
)

st.divider()

# Input area
review = st.text_area("✍️ Enter Movie Review", height=150)

col1, col2 = st.columns(2)

predict = col1.button(" Predict Sentiment")

if predict:

    if review.strip() == "":
        st.warning("⚠️ Please enter a movie review")
    
    else:

        review_vec = vectorizer.transform([review])

        prediction = model.predict(review_vec)
        decision = model.decision_function(review_vec)

        confidence = round(abs(decision[0]) * 10, 2)

        st.divider()

        # Sentiment result
        if prediction[0] == 1:

            st.markdown(
                f'<div class="result-box" style="background-color:#16a34a;color:white;">😊 Positive Review</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f'<div class="result-box" style="background-color:#dc2626;color:white;">😡 Negative Review</div>',
                unsafe_allow_html=True
            )

        st.write("")

        # Confidence
        st.metric("Prediction Confidence", f"{confidence}%")

        st.progress(min(confidence/100,1.0))

        st.write("")

        # Review stats
        st.subheader(" Review Analysis")

        words = review.split()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Words", len(words))
        col2.metric("Characters", len(review))
        col3.metric("Average Word Length", round(np.mean([len(w) for w in words]),2))

st.divider()

