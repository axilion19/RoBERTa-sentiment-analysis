import streamlit as st
import requests

st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")

st.markdown(
    """
    <style>
        .app-title {
            display: flex;              
            align-items: center;        
            gap: .5rem;                 
            font-size: 2.5rem;          
            font-weight: 700;
            margin: 0 0 1rem 0;        
        }
    </style>

    <div class="app-title">
        <span>🎬</span>
        <span>Movie Review Sentiment Analyzer</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("RoBERTa based sentiment analysis model for movie reviews.")

text = st.text_area("✍️ Write a review:", height=200)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Write a review before clicking the button.")
    else:
        with st.spinner("Running..."):
            try:
                response = requests.post("http://localhost:8000/predict", json={"text": text})
                result = response.json()

                label = result["prediction"]
                score = result["score"]

                if label == "POSITIVE":
                    st.success(f"😊 **Positive** review (Confidence: {score*100:.1f})")
                else:
                    st.error(f"😞 **Negative** review (Confidence: {score*100:.1f})")

            except Exception as e:
                st.error(f"Error: {e}")
