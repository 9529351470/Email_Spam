import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import time

ps = PorterStemmer()

# ------------------------
#    UI DESIGN STYLING
# ------------------------
st.set_page_config(page_title="Spam Classifier", layout="centered", page_icon="📩")

st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
        }
        .main-title {
            font-size: 40px;
            text-align: center;
            color: #4A4A4A;
            font-weight: 800;
            margin-bottom: 10px;
            animation: fadeIn 1s;
        }
        .sub-text {
            text-align: center;
            color: #6A6A6A;
            font-size: 18px;
            margin-bottom: 30px;
        }
        .result-good {
            padding: 20px;
            font-size: 28px;
            background-color: #d4f8d4;
            color: #1B8A1B;
            border-radius: 12px;
            text-align: center;
            font-weight: 700;
            animation: slideIn 0.5s;
        }
        .result-bad {
            padding: 20px;
            font-size: 28px;
            background-color: #ffd6d6;
            color: #C30000;
            border-radius: 12px;
            text-align: center;
            font-weight: 700;
            animation: slideIn 0.5s;
        }
        .confidence-bar {
            margin-top: 15px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 8px;
        }
        .stats-box {
            padding: 15px;
            background-color: #e8f4f8;
            border-radius: 10px;
            margin: 10px 0;
        }
        .stButton>button {
            width: 100%;
            background-color: #4A90E2;
            color: white;
            padding: 12px;
            font-size: 20px;
            font-weight: 600;
            border-radius: 10px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #1664c4;
            transform: scale(1.02);
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .feature-highlight {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------
#    SESSION STATE INIT
# ------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_checked' not in st.session_state:
    st.session_state.total_checked = 0
if 'spam_count' not in st.session_state:
    st.session_state.spam_count = 0


# ------------------------
#    TEXT TRANSFORMATION
# ------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# ------------------------
#    LOAD MODEL & TFIDF
# ------------------------
@st.cache_resource
def load_models():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model


tfidf, model = load_models()

# ------------------------
#       MAIN UI
# ------------------------
st.markdown('<div class="main-title">📩 Email / SMS Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Detect whether your message is safe or spam using Machine Learning.</div>',
            unsafe_allow_html=True)

# Sidebar with additional features
with st.sidebar:
    st.header("📊 Statistics")
    st.metric("Total Messages Checked", st.session_state.total_checked)
    st.metric("Spam Detected", st.session_state.spam_count)
    if st.session_state.total_checked > 0:
        spam_rate = (st.session_state.spam_count / st.session_state.total_checked) * 100
        st.metric("Spam Rate", f"{spam_rate:.1f}%")

    st.markdown("---")
    st.header("🎯 Quick Examples")
    if st.button("Try Spam Example"):
        st.session_state.example_text = "WINNER!! You have won a $1000 prize! Click here now to claim your reward!"
    if st.button("Try Normal Example"):
        st.session_state.example_text = "Hi, let's meet for coffee tomorrow at 3pm. Looking forward to catching up!"

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.total_checked = 0
        st.session_state.spam_count = 0
        st.rerun()

# Main input area
input_sms = st.text_area(
    "✏️ Enter your message here",
    height=150,
    value=st.session_state.get('example_text', ''),
    placeholder="Type or paste your message here..."
)

# Clear the example text after it's used
if 'example_text' in st.session_state:
    del st.session_state.example_text

col1, col2 = st.columns([3, 1])
with col1:
    predict_button = st.button("🔍 Predict")
with col2:
    clear_button = st.button("🔄 Clear")

if clear_button:
    st.rerun()

if predict_button:
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message!")
    else:
        # Show loading animation
        with st.spinner('🤖 Analyzing message...'):
            time.sleep(0.5)  # Brief pause for effect

            # 1. preprocess
            transformed_sms = transform_text(input_sms)

            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])

            # 3. predict
            result = model.predict(vector_input)[0]

            # Get probability if available
            try:
                probabilities = model.predict_proba(vector_input)[0]
                confidence = max(probabilities) * 100
            except:
                confidence = None

            # Update statistics
            st.session_state.total_checked += 1
            if result == 1:
                st.session_state.spam_count += 1

            # Add to history
            st.session_state.history.insert(0, {
                'message': input_sms[:50] + '...' if len(input_sms) > 50 else input_sms,
                'result': 'Spam' if result == 1 else 'Not Spam',
                'confidence': confidence
            })
            # Keep only last 5 entries
            st.session_state.history = st.session_state.history[:5]

        # 4. Display result
        st.markdown("---")
        if result == 1:
            st.markdown('<div class="result-bad">🚫 Spam Detected!</div>', unsafe_allow_html=True)
            st.error(
                "⚠️ This message appears to be spam. Be cautious about clicking links or sharing personal information.")
        else:
            st.markdown('<div class="result-good">✅ Not Spam</div>', unsafe_allow_html=True)
            st.success("✨ This message appears to be legitimate.")

        # Display confidence
        if confidence:
            st.markdown(f"**Confidence Level:** {confidence:.1f}%")
            st.progress(confidence / 100)

        # Message analysis
        with st.expander("📝 Message Analysis"):
            word_count = len(input_sms.split())
            char_count = len(input_sms)

            col1, col2, col3 = st.columns(3)
            col1.metric("Words", word_count)
            col2.metric("Characters", char_count)
            col3.metric("Processed Tokens", len(transformed_sms.split()))

            st.markdown("**Processed Text:**")
            st.code(transformed_sms)

# Display history
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Recent Checks")
    for idx, item in enumerate(st.session_state.history, 1):
        with st.expander(f"{idx}. {item['message']} - **{item['result']}**"):
            if item['confidence']:
                st.write(f"Confidence: {item['confidence']:.1f}%")

