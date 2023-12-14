import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import plotly.graph_objects as go
from math import pi
import matplotlib.pyplot as plt


# Load Data
@st.cache_resource
def load_dataset():
    try:
        return pd.read_csv('data/df_processed.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_dataset()



# Creating TF-IDF Matrix
@st.cache_data
def create_tfidf_matrix(df):
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
        return tfidf_matrix
    except Exception as e:
        st.error(f"Error in TF-IDF Matrix creation: {e}")
        return None

tfidf_matrix = create_tfidf_matrix(df)


# Sample function (customize as needed)
def calculate_similarity_between_two_songs(df, tfidf_matrix, input_index_1, input_index_2, weights, selected_features):
    # ... [Rest of the function remains unchanged] ...
    # Check if the indices are within the range of the DataFrame
    # if input_index_1 >= len(df) or input_index_2 >= len(df):
    #     return "One or both indices are out of range.", {}

    # Find integer indices of selected songs
    idx1 = df[df['song'] == input_index_1].index[0]
    idx2 = df[df['song'] == input_index_2].index[0]

    # Extracting features of both songs
    song_1 = df.iloc[idx1]
    song_2 = df.iloc[idx2]

    # Calculate textual similarity
    lyric_similarity = cosine_similarity(tfidf_matrix[idx1], tfidf_matrix[idx1])[0][0]

    # Convert chord progressions to strings and calculate similarity
    chord_strings = [' '.join(str(cp).split(',')) for cp in df['cp']]
    vectorizer = CountVectorizer()
    chord_vectors = vectorizer.fit_transform(chord_strings)
    chord_similarity = cosine_similarity(chord_vectors[idx1], chord_vectors[idx1])[0][0]

    numerical_similarity = sum(
        weights.get(feature, 0) * (1 - abs(song_1[feature] - song_2[feature]) / df[feature].max())
        for feature in selected_features if feature not in ['lyrics_similarity', 'chord_similarity']
    ) / sum(weights.values())

    # Only add the weights of selected similarity features
    combined_similarity = numerical_similarity
    if 'lyrics_similarity' in selected_features:
        combined_similarity += weights.get('lyrics', 0) * df.loc[idx1, 'lyrics_similarity']
    if 'chord_similarity' in selected_features:
        combined_similarity += weights.get('cp', 0) * df.loc[idx1, 'chord_similarity']

    song_details = {
        'song_1': {
            'title': song_1['song'],
            'key': song_1['key_labels'],
            'lyrics': song_1['text_2'][:500] + "..."
        },
        'song_2': {
            'title': song_2['song'],
            'key': song_2['key_labels'],
            'lyrics': song_2['text_2'][:500] + "..."
        }
    }

    return combined_similarity, song_details

# Plot Radar Chart using Plotly
def plot_radar_chart_plotly(df, idx1, idx2, features):
    # Create a list of maximum values for each feature for normalization
    max_values = df[features].max().tolist()

    # Retrieve the feature values for each song
    song1_values = df.iloc[idx1][features].tolist()
    song2_values = df.iloc[idx2][features].tolist()

    # Normalize the feature values by dividing by the maximum value
    song1_normalized = [float(i)/max(j, 1) for i, j in zip(song1_values, max_values)]  # max(j, 1) to avoid division by zero
    song2_normalized = [float(i)/max(j, 1) for i, j in zip(song2_values, max_values)]

    # Calculate chord and textual similarity as before and add them to the normalized lists
    # Assuming these similarities are already between 0 and 1
    chord_strings = [' '.join(str(cp).split(',')) for cp in df['cp']]
    vectorizer = CountVectorizer()
    chord_vectors = vectorizer.fit_transform(chord_strings)
    chord_similarity = cosine_similarity(chord_vectors[idx1], chord_vectors[idx1])[0][0]
    lyric_similarity = cosine_similarity(tfidf_matrix[idx1], tfidf_matrix[idx2])[0][0]

    song1_normalized += [chord_similarity, lyric_similarity]
    song2_normalized += [chord_similarity, lyric_similarity]
    
    # Add chord and textual similarity to the categories
    categories = features + ['Chord Similarity', 'Textual Similarity']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=song1_normalized,
        theta=categories,
        fill='toself',
        name='Song 1'
    ))

    fig.add_trace(go.Scatterpolar(
        r=song2_normalized,
        theta=categories,
        fill='toself',
        name='Song 2'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)



    # Add a legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Show the plot
    # st.pyplot(fig)

# Streamlit UI
def show():
    st.title('Song Similarity Calculator')

    # Sidebar for Feature Selection
    selected_features = st.sidebar.multiselect(
        'Select Features for Comparison', 
        ['danceability', 'energy', 'key', 'loudness', 'speechiness', 
         'acousticness', 'instrumentalness', 'liveness', 'valence', 
         'tempo', 'time_signature', 'lyrics_similarity', 'chord_similarity'], 
        default=['danceability', 'energy', 'key', 'loudness', 'speechiness', 
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 
                 'tempo', 'time_signature']
    )

    # Dropdown for Song Selection
    song_names = df['song'].unique()
    input_index_1 = st.selectbox('Select Song 1', song_names, index=0)
    input_index_2 = st.selectbox('Select Song 2', song_names, index=1)

    # Find integer indices of selected songs
    idx1 = df[df['song'] == input_index_1].index[0]
    idx2 = df[df['song'] == input_index_2].index[0]

    

    # User input for weights
    st.sidebar.header('Feature Weights')
    weights = {
    'lyrics': st.sidebar.slider('Lyrics Weight', 0.0, 1.0, 0.4),
    'cp': st.sidebar.slider('Chord Progression Weight', 0.0, 1.0, 0.05),
    'danceability': st.sidebar.slider('Danceability Weight', 0.0, 1.0, 0.05),
    'energy': st.sidebar.slider('Energy Weight', 0.0, 1.0, 0.05),
    'key': st.sidebar.slider('Key Weight', 0.0, 1.0, 0.05),
    'loudness': st.sidebar.slider('Loudness Weight', 0.0, 1.0, 0.05),
    'speechiness': st.sidebar.slider('Speechiness Weight', 0.0, 1.0, 0.05),
    'acousticness': st.sidebar.slider('Acousticness Weight', 0.0, 1.0, 0.05),
    'instrumentalness': st.sidebar.slider('Instrumentalness Weight', 0.0, 1.0, 0.05),
    'liveness': st.sidebar.slider('Liveness Weight', 0.0, 1.0, 0.05),
    'valence': st.sidebar.slider('Valence Weight', 0.0, 1.0, 0.05),
    'tempo': st.sidebar.slider('Tempo Weight', 0.0, 1.0, 0.05),
    'time_signature': st.sidebar.slider('Time Signature Weight', 0.0, 1.0, 0.05)
}

    if st.button('Calculate Similarity'):
        similarity_score, song_details = calculate_similarity_between_two_songs(df, tfidf_matrix, input_index_1, input_index_2, weights, selected_features)
        chord_strings = [' '.join(str(cp).split(',')) for cp in df['cp']]
        vectorizer = CountVectorizer()
        chord_vectors = vectorizer.fit_transform(chord_strings)
        chord_similarity = cosine_similarity(chord_vectors[idx1], chord_vectors[idx1])[0][0]
        lyric_similarity = cosine_similarity(tfidf_matrix[idx1], tfidf_matrix[idx1])[0][0]

        # Store the similarities in the DataFrame for use in feature selection and radar chart
        df.loc[idx1, 'lyrics_similarity'] = lyric_similarity
        df.loc[idx2, 'lyrics_similarity'] = lyric_similarity
        df.loc[idx1, 'chord_similarity'] = chord_similarity
        df.loc[idx2, 'chord_similarity'] = chord_similarity

        # Calculate combined similarity using the selected features
        similarity_score, song_details = calculate_similarity_between_two_songs(
            df, tfidf_matrix, input_index_1, input_index_2, weights, selected_features
        )
        
        # Radar chart plot
        feature_labels = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
        plot_radar_chart_plotly(df, idx1, idx2, selected_features)


        if isinstance(similarity_score, str):  # Error message returned
            st.error(similarity_score)
        else:
    # Display the similarity score in bold, large, red font
            st.markdown(f'<h2 style="color:red;">Similarity Score: {similarity_score:.4f}</h2>', unsafe_allow_html=True)
  

            
            # Displaying song details
            st.subheader('Song 1 Details:')
            st.write(f"**Title:** {song_details['song_1']['title']}")
            st.write(f"**Key:** {song_details['song_1']['key']}")
            st.write(f"**Lyrics (Snippet):** {song_details['song_1']['lyrics']}")

            st.subheader('Song 2 Details:')
            st.write(f"**Title:** {song_details['song_2']['title']}")
            st.write(f"**Key:** {song_details['song_2']['key']}")
            st.write(f"**Lyrics (Snippet):** {song_details['song_2']['lyrics']}")

    # Set page background style
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://miro.medium.com/v2/resize:fit:1400/1*VOpNR4SysBXTHbpIpTHjkQ.jpeg");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    show()
