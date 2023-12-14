import streamlit as st
from PIL import Image
import pathlib
import logging
import shutil



# Configuring page
st.set_page_config(page_title='Song Feature Explorer', page_icon=':bar_chart:', layout='wide')

# Disclaimer Section
st.warning("""
           **Disclaimer:** 
           
           The data in this app comes from two public datasets: the [Chord Progression Assistant](https://github.com/jhamer90811/chord_progression_assistant) and [the Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset). These datasets provide a limited perspective and do not cover the full spectrum of music data. The analyses and insights offered by this app may not fully represent the music landscape and could contain inherent biases. Users are advised to consider these limitations when interpreting the app's outputs.
""")

# App Title
st.title("Music Feature Explorer and Song Similarity Calculator")



# Project Context
st.subheader('Project Context')
st.write("""
    Music transcends mere auditory enjoyment, acting as a powerful catalyst for emotions, memories, and collective sentiments. The intricate dance of lyrics and chord progressions reveals the essence of a song's ambiance. This synergy is crucial not only for academic study but also for record labels seeking bands that resonate with diverse audiences. At the heart of this exploration is the relationship between lyrics and the emotions they evoke. Understanding the complex tapestry of music, however, extends beyond subjective interpretation and into the realm of quantitative analysis.
    
    This project employs visualizations to explore numerical features like tempo, key, and chord progression, alongside textual elements. By quantifying these aspects, we uncover patterns and trends in music. Additionally, radar graphs offer a nuanced understanding of the musical and lyrical kinship between songs, aiding both academic and practical applications in the music industry.
    
    Ultimately, this project aims to unravel the science behind the art of music, shedding light on how various elements of a song coalesce to impact its listeners.
""")

# Target Audience
st.subheader('Target Audience')
st.write("""
    Our primary audience includes Record Labels, Musicians, and AI-Powered Music Creation platforms.
    
    - **Record Labels**: They benefit from data-driven insights into music trends and listener preferences, aiding in scouting and signing artists whose music resonates emotionally.
    
    - **Musicians**: Artists can explore music to find inspiration, understand patterns, and stay connected with evolving trends.
    
    - **AI-Powered Music Creation**: This sector can use our rich data repository to develop algorithms for creating targeted music content, aligning with specific moods and themes.
    
    This app serves those at the intersection of technology, creativity, and business in the music industry, providing insights into the interplay between musical elements and emotional impact.
""")

# Key Variables
st.subheader('Key Variables')
st.write("""
    Several key variables are crucial in understanding a track's appeal. Danceability, energy, instrumentalness, key, liveness, loudness, tempo, duration, time signature, and valence each play a role in the character of a song. These variables, along with chord progression and textual analysis, help in exploring musical patterns and emotional resonance. Our app visualizes these aspects to offer a comprehensive view of a song's characteristics.
""")

# App Outline
st.subheader('Outline')
st.write("""
    **1. Numerical Feature Exploration Page:** Users can select variables like key, tempo, and danceability for exploration, visualized through bar charts, pie charts, and ridge plots.
    
    **2. Textual Feature Exploration Page:** This page offers bubble word clouds and LDA word clouds, allowing users to delve into the thematic and emotional content of lyrics.
    
    **3. Song Similarity Feature Calculator Page:** Users can calculate and visualize the similarity between two songs using a sophisticated algorithm and radar graphs.
    
    These features are designed for a range of users, from casual listeners to industry professionals, to understand and explore the multifaceted world of music.
""")

# Footer with Contact Information
c1, c2, c3 = st.columns(3)
with c1:
    st.info('**Data Analyst: [Personal Website](https://oliz888.github.io/)**', icon="💡")
with c2:
    st.info('**GitHub: [@Oliz888](https://github.com/Oliz888)**', icon="💻")
with c3:
    st.info('**Project Info: [Music Recommendation](https://music-recommendation.my.canva.site/)**', icon="🧠")



