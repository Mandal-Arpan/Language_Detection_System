import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Language Detection App",
    page_icon="🌍",
    layout="centered"
)

st.image('pic.png', use_container_width=True)
# -------------------------------------------------
# Load Model & Tokenizer
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("saved_model/simple_rnn_model.h5")
    with open("saved_model/tokenizer.pkl", "rb") as f:
        tokenizer, label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_artifacts()

# -------------------------------------------------
# Prediction Function
# -------------------------------------------------
def predict_language(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=80)
    probs = model.predict(padded, verbose=0)[0]
    class_index = np.argmax(probs)
    language = label_encoder.inverse_transform([class_index])[0]
    confidence = probs[class_index]
    return language, confidence

# -------------------------------------------------
# Title & Description
# -------------------------------------------------
st.title(":violet[🌍 Language Detection System]")
st.caption(
    "Enter a sentence and the model will predict the language using a trained RNN."
)

st.markdown("---")

# -------------------------------------------------
# Input Section
# -------------------------------------------------
st.subheader("✍️ Enter Text")

user_text = st.text_area(
    "Text Input",
    height=130,
    placeholder="Example:\n- This is a beautiful day\n- यह एक अच्छा दिन है\n- Ceci est une belle journée"
)

# Optional helper examples
with st.expander("💡 Example Sentences"):
    st.markdown(
        """
        - **English:** This is a beautiful day  
        - **Hindi:** यह एक अच्छा दिन है  
        - **French:** Ceci est une belle journée  
        - **Spanish:** Este es un buen día  
        """
    )

st.markdown("---")

# -------------------------------------------------
# Prediction Button & Output
# -------------------------------------------------
if st.button("🔍 Detect Language", use_container_width=True):

    if user_text.strip() == "":
        st.warning("Please enter some text to detect the language.")
    else:
        with st.spinner("Analyzing text..."):
            language, confidence = predict_language(user_text)

        st.subheader("🔎 Prediction Result")
        st.success(f"**Predicted Language:** {language}")
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.3f}"
        )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(" 2026 Language Detection App | RNN • TensorFlow • Streamlit")
