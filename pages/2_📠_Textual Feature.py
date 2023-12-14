# Libraries
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans  
from sklearn.feature_extraction.text import TfidfVectorizer

# Config
st.set_page_config(page_title='Textual Feature', page_icon=':bar_chart:', layout='wide')

# Load Data
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('data/df_processed.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_dataset()

# Title
st.title('📠 Textual Feature Explorer')

# Add a sidebar to customize WordCloud parameters
st.sidebar.title("Word Cloud Options")
max_words = st.sidebar.slider("Max Words", 10, 1000, 400)
max_font_size = st.sidebar.slider("Max Font Size", 10, 100, 60)

# Display the WordCloud based on user-selected parameters
st.write("WordCloud with Custom Parameters")

def generate_wordcloud(data, max_words, max_font_size):
    wordcloud = WordCloud(
        background_color='black',
        max_words=max_words,
        max_font_size=max_font_size,
        scale=5,
        random_state=42
    ).generate(str(data))

    plt.figure(figsize=(20, 15))
    plt.axis('off')
    plt.imshow(wordcloud)
    st.pyplot(plt)

generate_wordcloud(df['cleaned_text'], max_words, max_font_size)



