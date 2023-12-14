import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Title
st.title('🔊 Numerical Feature Explorer')

# Load Data
@st.cache_data()
def load_dataset():
    try:
        return pd.read_csv('data/df_processed.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_dataset()

# Categorize tempo for Pie Chart
def categorize_tempo(tempo):
    if tempo < 120:
        return "<120"
    elif 120 <= tempo <= 156:
        return "120-156"
    elif 156 < tempo <= 176:
        return "156-176"
    else:
        return ">176"

df['tempo_category'] = df['tempo'].apply(categorize_tempo)

# Interactive Pie Chart
st.sidebar.header("Pie Chart Settings")
pie_chart_features = ['key', 'tempo_category', 'label']  
selected_pie_chart_feature = st.sidebar.selectbox("Select feature for pie chart", pie_chart_features)

# For displaying the 'key_label' instead of 'key'
if selected_pie_chart_feature == 'key':
    fig_pie = px.pie(df, names='key_labels', title=f'Distribution of Key Labels')
else:
    fig_pie = px.pie(df, names=selected_pie_chart_feature, title=f'Distribution of {selected_pie_chart_feature}')

st.plotly_chart(fig_pie, use_container_width=True)





# # Interactive Bar Plot
# st.sidebar.header("Bar Plot Settings")
# bar_plot_x_features = ['key', 'label', 'tempo_category']  # Updated categories
# bar_plot_y_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
# selected_bar_x_feature = st.sidebar.selectbox("Select X-axis for bar plot", bar_plot_x_features)
# selected_bar_y_feature = st.sidebar.selectbox("Select Y-axis for bar plot", bar_plot_y_features)

# # Calculate the mean of the selected y-axis variable for each category of the selected x-axis variable
# mean_y_values = df.groupby(selected_bar_x_feature)[selected_bar_y_feature].mean().reset_index()

# # Create a subtitle indicating the selected y-axis variable
# subtitle = f"Mean {selected_bar_y_feature.capitalize()} by {selected_bar_x_feature.capitalize()}"

# # Create a custom x-axis and y-axis labels based on user selection
# x_label = f"{selected_bar_x_feature.capitalize()}"
# y_label = f"{selected_bar_y_feature.capitalize()} Mean (Rounded to 0.01)"

# # Round up the y-axis values to the nearest 0.01
# mean_y_values[selected_bar_y_feature] = np.round(mean_y_values[selected_bar_y_feature], 2)

# fig_bar = px.bar(mean_y_values, x=selected_bar_x_feature, y=selected_bar_y_feature,
#                  title=f'Distribution of {selected_bar_y_feature} by {selected_bar_x_feature}',
#                  text=selected_bar_y_feature)  # Add values as text on bars

# # Add the subtitle to the top-left corner of the plot
# fig_bar.add_annotation(
#     text=subtitle,
#     xref="paper", yref="paper",
#     x=0.02, y=1.0,
#     showarrow=False,
#     font=dict(size=12)
# )

# # Set the custom x-axis and y-axis labels
# fig_bar.update_xaxes(title_text=x_label)
# fig_bar.update_yaxes(title_text=y_label)

# # Add a source with the weighting mechanism description at the bottom-right
# source_text = "Source: Data processed with a weighting mechanism to calculate mean values."
# fig_bar.add_annotation(
#     text=source_text,
#     xref="paper", yref="paper",
#     x=0.98, y=-0.1,
#     showarrow=False,
#     font=dict(size=10),
#     align="right"
# )

# # Adjust the y-axis position to avoid overlapping
# fig_bar.update_layout(yaxis=dict(domain=[0.15, 0.9]))

# st.plotly_chart(fig_bar, use_container_width=True)


# Interactive Bar Plot
st.sidebar.header("Bar Plot Settings")
bar_plot_x_features = ['key', 'label', 'tempo_category']  # Updated categories
bar_plot_y_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
selected_bar_x_feature = st.sidebar.selectbox("Select X-axis for bar plot", bar_plot_x_features)
selected_bar_y_feature = st.sidebar.selectbox("Select Y-axis for bar plot", bar_plot_y_features)

# Calculate the mean of the selected y-axis variable for each category of the selected x-axis variable
mean_y_values = df.groupby(selected_bar_x_feature)[selected_bar_y_feature].mean().reset_index()

# Create a subtitle indicating the selected y-axis variable
subtitle = f"Mean {selected_bar_y_feature.capitalize()} by {selected_bar_x_feature.capitalize()}"

# Create a custom x-axis and y-axis labels based on user selection
x_label = f"{selected_bar_x_feature.capitalize()}"
y_label = f"{selected_bar_y_feature.capitalize()} Mean (Rounded to 0.01)"

# Round up the y-axis values to the nearest 0.01
mean_y_values[selected_bar_y_feature] = np.round(mean_y_values[selected_bar_y_feature], 2)

# Define a color variable based on the selected y-axis variable values
color_variable = mean_y_values[selected_bar_y_feature]

fig_bar = px.bar(mean_y_values, x=selected_bar_x_feature, y=selected_bar_y_feature,
                 title=f'Distribution of {selected_bar_y_feature} by {selected_bar_x_feature}',
                 text=selected_bar_y_feature, color=color_variable, color_continuous_scale='viridis')  # Color by values

# Add the subtitle to the top-left corner of the plot
fig_bar.add_annotation(
    text=subtitle,
    xref="paper", yref="paper",
    x=0.02, y=1.0,
    showarrow=False,
    font=dict(size=12)
)

# Set the custom x-axis and y-axis labels
fig_bar.update_xaxes(title_text=x_label)
fig_bar.update_yaxes(title_text=y_label)

# Add a source with the weighting mechanism description at the bottom-right
source_text = "Source: Data processed with a weighting mechanism to calculate mean values."
fig_bar.add_annotation(
    text=source_text,
    xref="paper", yref="paper",
    x=0.98, y=-0.1,
    showarrow=False,
    font=dict(size=10),
    align="right"
)

# Adjust the y-axis position to avoid overlapping
fig_bar.update_layout(yaxis=dict(domain=[0.15, 0.9]))

st.plotly_chart(fig_bar, use_container_width=True)