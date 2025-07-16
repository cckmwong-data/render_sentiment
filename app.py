import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
from wordcloud import WordCloud

from tensorflow.keras.models import load_model
import pickle

import streamlit as st
import joblib
import os
import string
import re
import streamlit.components.v1 as components
import html

from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from tensorflow.keras.models import load_model

from googleapiclient.discovery import build

nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

# Download NLTK assets to that folder
#nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
#nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
#nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)

# Load the trained LSTM model
#model = load_model('sentiment_lstm_model.h5')

MODEL_PATH = "sentiment_lstm_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
    
# Load max_length
with open('max_length_sentiment.txt', 'r') as f:
    max_length = int(f.read())

# Load the tokenizer
with open('tokenizer_sentiment.json', 'r') as f:
    data = f.read()  # Read the file content as a string
tokenizer = tokenizer_from_json(data)  # Pass the JSON string to the function

def remove_mention_url_email(text):
    # remove mentions, URLs, and emails by replacing these patterns by space
    # and then change to lower case
    text = re.sub(r"@\S+|https?:\S+|http?:\S+|\S+@\S+", ' ', str(text).lower())

    # remove extra spaces
    return text.strip()

def remove_html_tags(text):
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)

    # Decode HTML entities
    return html.unescape(text)

# remove punctuation and other non-alphabetic characters
def remove_punc(text):
    # replace anything that is NOT a lowercase letter or space to space
    text = re.sub(r"[^a-z\s]", ' ', text)
    return text

# function to convert nltk POS (part of speech) tag to WordNet POS tag
def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# change to lemmatized text with the consideration of the POS
def lemmatize_text(words):
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(words) # get NLTK’s pos_tag
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag))
        for word, pos_tag in pos_tags
    ]
    return lemmatized_words

# function to remove stopwords
def remove_stopwords(words):
    tokens = []
    stop_words = stopwords.words('english')

    # only keep those words which are not stop words with length greater than 2
    for word in words:
      if len(word) >= 2 and word not in stop_words:
        tokens.append(word)

    return tokens

# Full preprocessing and prediction
def preprocess_and_predict(text):
    tokenizer_nltk = TweetTokenizer()
    text = remove_html_tags(text)
    text = remove_mention_url_email(text)
    text = remove_punc(text)
    tokens = tokenizer_nltk.tokenize(text)
    tokens = lemmatize_text(tokens)
    tokens = remove_stopwords(tokens)

    seq = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')

    prob = model.predict(padded)[0][0]
    #sentiment = 'Positive' if prob > 0.5 else 'Negative'
    if prob >= 0.55:
        sentiment = 'Positive'
    elif prob <= 0.45:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
        
    return sentiment, float(prob), tokens

from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    parsed_url = urlparse(url)
    if 'youtu.be' in parsed_url.netloc:
        return parsed_url.path.strip('/')
    elif 'youtube.com' in parsed_url.netloc:
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif '/embed/' in parsed_url.path:
            return parsed_url.path.split('/embed/')[1]
    return None


def fetch_comments(user_input):
    #from dotenv import load_dotenv
    #import os
    #load_dotenv()  # load .env file

    yt_api_key = os.getenv("YOUTUBE_API_KEY")
    #yt_api_key = st.secrets["YOUTUBE_API_KEY"]

    # API setup
    api_key = yt_api_key
    youtube = build('youtube', 'v3', developerKey=api_key)

    video_id = extract_video_id(user_input)

    #Fetch comments
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()

        # Loops through each comment returned
        # Extracts the actual comment text (textDisplay)
        # Appends it to your comments list

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        #If there’s another page of comments, next_page_token will exist
        #If not, the loop breaks — all pages are done

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    return comments


def generate_wordcloud(df, sentiment, colour):
    # Filter only spam messages
    words = df['tokens']

    # If 'Message' is a list of tokens, join them into strings
    if isinstance(words.iloc[0], list):
        words = words.apply(lambda x: ' '.join(x))

    # Combine all spam messages into one string
    text = ' '.join(words)

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, colormap=colour).generate(text)

    # Plot the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Most Frequent Words of {sentiment} Messages", fontsize=16)
    st.pyplot(fig)


def show_pie_chart(df):
    # Count the number of each sentiment
    sentiment_counts = df['sentiment'].value_counts()

    colors = ['#4DB6AC', '#81D4FA', '#B39DDB']  # teal, sky blue, purple

    # Pie chart
    plt.pie(sentiment_counts, 
            labels=sentiment_counts.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            explode=[0.05]*len(sentiment_counts),
            colors = colors)

    plt.title("Sentiment Distribution of Comments")
    plt.axis('equal')  # Make it a circle
    plt.show()

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# UI
st.markdown("<h1 style='font-size: 54px;'>Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 20px;'>This tool is for analyzing sentiment of Youtube comments. </p>", unsafe_allow_html=True)

st.markdown("<label style='font-size: 18px;'>Please copy and paste the YouTube link here:</label>", unsafe_allow_html=True)
# Text input area
user_input = st.text_area("", key="user_input")

if st.button("🔍Analyze Comments", use_container_width=True):  
    try:
        comments = fetch_comments(user_input)
        
        df = pd.DataFrame(comments, columns=["comment"])
        df_comments = pd.DataFrame()

        # Initialize progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()  # Placeholder for progress text

        sentiment = []
        score = []
        tokens = []

        total_comments = len(df)

        for i, row in df.iterrows():
            comment = row['comment']
            s, sc, t = preprocess_and_predict(comment)
            sentiment.append(s)
            score.append(sc)
            tokens.append(t)

            # Update progress bar
            progress = (i + 1) / total_comments
            progress_bar.progress(progress)  # Update progress bar
            progress_text.text(f"Processing comment {i + 1} of {total_comments}...")

        # Add to DataFrame
        df['tokens'] = tokens
        df['sentiment'] = sentiment
        df['score'] = score

        # Store processed data in session state
        st.session_state["df"] = df
        st.session_state["df_positive"] = df[df.sentiment == "Positive"]
        st.session_state["df_negative"] = df[df.sentiment == "Negative"]

        #df_comments = df[['comment', 'sentiment']]
        # prepare csv for download
        st.session_state["csv"] = convert_df(df)

        # Clear progress bar
        progress_bar.empty()
        progress_text.empty()

        # Display results if data exists in session state
        if "df" in st.session_state:
            df = st.session_state["df"]
            df_positive = st.session_state["df_positive"]
            df_negative = st.session_state["df_negative"]

            # CAUTION!!!!! Download Button
            st.download_button(
            label="Download All Comments",
            data=st.session_state["csv"],  # Use cached CSV data
            file_name="comments.csv",
            mime="text/csv",
            key="download-csv"
            )

            st.write(f"Number of comments downloaded: {len(df)}")
            st.write(f"Number of positive comments: {len(df_positive)}")
            st.write(f"Number of negative comments: {len(df_negative)}")

            # Dropdown to choose sentiment (optional)
            #user_sentiment = st.selectbox("Select Sentiment for Word Cloud Generator:", ["Positive", "Negative"], index=0)

            generate_wordcloud(df_positive, "Positive", "viridis")
            generate_wordcloud(df_negative, "Negative", "cividis")
    except Exception as e:
        st.warning(f"⚠️ Please enter a valid Youtube link.")