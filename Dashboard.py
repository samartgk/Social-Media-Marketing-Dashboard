import os
import seaborn as sns
import streamlit as st
import matplotlib
import random
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import gc
import math
import pandas as pd
import numpy as np
import base64
from PIL import Image


def Set_Background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


@st.cache_data
def load_data():
    data = pd.read_csv(r"social_media_usage.csv")
    data["Engagement_Rate"] = (data["Likes_Per_Day"] + data["Follows_Per_Day"]) / data["Posts_Per_Day"]
    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data


Set_Background(r"new.png")

df = load_data()



st.sidebar.title("Filters")
st.sidebar.info("Select platforms here to apply filters.")
platforms = df["App"].unique()
selected_platforms = st.sidebar.multiselect("Select Platform(s)", platforms, default=platforms)


df_filtered = df[df["App"].isin(selected_platforms)]


st.markdown(
    "<h1 style='text-align: center; color: red; font-size: 3em;'>Social Media Marketing Insights Dashboard</h1>",
    unsafe_allow_html=True)


st.subheader("Key Performance Indicators (KPIs)")
col1, col2, col3 = st.columns(3)
col1.metric("Avg. Daily Minutes", round(df_filtered["Daily_Minutes_Spent"].mean(), 2))
col2.metric("Avg. Posts Per Day", round(df_filtered["Posts_Per_Day"].mean(), 2))
col3.metric("Avg. Engagement Rate", round(df_filtered["Engagement_Rate"].mean(), 2))



def transparent_bg(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig



st.subheader("User Distribution Across Platforms")
fig1 = px.pie(df_filtered, names="App", title="Users per Platform")
st.plotly_chart(transparent_bg(fig1))


st.subheader("Daily Minutes Spent Per Platform")
fig2 = px.bar(df_filtered, x="App", y="Daily_Minutes_Spent", color="App", title="Time Spent on Platforms",
              barmode="group")
st.plotly_chart(transparent_bg(fig2))


st.subheader("Posts vs. Engagement Rate")
fig3 = px.scatter(df_filtered, x="Posts_Per_Day", y="Engagement_Rate", color="App",
                  size="Likes_Per_Day", hover_data=["User_ID"],
                  title="Posts vs. Engagement Rate")
st.plotly_chart(transparent_bg(fig3))


st.subheader("Follows Per Day by Platform")
fig4 = px.box(df_filtered, x="App", y="Follows_Per_Day", color="App", title="Variation in Follows Per Day")
st.plotly_chart(transparent_bg(fig4))


st.subheader("Correlation Heatmap (KPIs)")


numeric_df = df_filtered.drop(columns=["User_ID", "App"])
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax, cbar_kws={'label': 'Correlation'},
            annot_kws={'size': 12, 'weight': 'bold', 'color': 'white'})  # Change annotation color to white

ax.set_facecolor('none')
fig.patch.set_facecolor('none')

ax.set_xlabel(ax.get_xlabel(), color='white')
ax.set_ylabel(ax.get_ylabel(), color='white')
ax.tick_params(axis='x', labelcolor='white')
ax.tick_params(axis='y', labelcolor='white')
st.pyplot(fig)


st.subheader("Distribution of Daily Minutes Spent")
fig6 = px.histogram(df_filtered, x="Daily_Minutes_Spent", nbins=10, title="Engagement Time Distribution")
st.plotly_chart(transparent_bg(fig6))


st.subheader("Top 5 Users by Engagement Rate")
top_users = df_filtered.nlargest(5, "Engagement_Rate")
fig7 = px.bar(top_users, x="User_ID", y="Engagement_Rate", color="App", title="Most Engaged Users")
st.plotly_chart(transparent_bg(fig7))

with st.sidebar:
    with st.expander('About'):
        st.title("Welcome to Social Media Marketing Insights Dashboard!")

        st.write("A simple Dashboard to analyse and get insights into the dataset which contains usage data of "
                         "multiple users on different platforms to identify objectives useful for Marketing on "
                         "Social Media Platforms.")

        st.header("What this does:")
        st.write(
            """
            * Provides Valuable Insights into social media marketing.
            * Allows you to change visualisations based on the filters and drill-through.
            """
        )

        st.header("How to use it:")
        st.write(
            """
            1.  Navigate to the sidebar to select the required platforms.
            2.  For filtering on each individual Visualisation use the respective legend to select or unselect attributes.
            3.  Enjoy!
            """
        )

        st.header("Credits:")
        st.write(
            """
            The dataset was the Social Media Usage Dataset by bhadramohit: Available on Kaggle, this dataset allows 
            exploration of social media usage trends, aiding in decision-making for marketing strategies, 
            content creation, and platform engagement. \n
            Link: https://www.kaggle.com/datasets/bhadramohit/social-media-usage-datasetapplications?utm_source=chatgpt.com
            """
        )

        st.header("About the Developer:")
        st.write(
            """
            This app was created by ALPHA.
            """
        )
        image = Image.open(r"my_photo.jpg")
        resized_image = image.resize((100, 100))
        st.image(resized_image, caption="Developer")
