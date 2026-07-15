import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

st.set_page_config(page_title="Netflix Content Strategy Dashboard", layout="wide")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
 return pd.read_csv("data set/netflix_cleaned (1).csv")

df = load_data()

# ----------------------------
# Load Models
# ----------------------------

classifier = joblib.load("model.pkl")
clustering = joblib.load("model.pkl")
country_encoder = joblib.load("country_encoder.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")
# ----------------------------
# Prepare Data
# ----------------------------
df["duration_num"] = df["duration"].astype(str).str.extract(r"(\d+)").astype(float)

df["country"] = df["country"].fillna("India")
df["country"] = df["country"].apply(lambda x: x if x in country_encoder.classes_ else "India")
df["country_encoded"] = country_encoder.transform(df["country"])
x = pd.DataFrame({
    "country": df["country_encoded"],
 "listed_in": df["listed_in"].str.split(",").str[0].str.strip().apply(
    lambda x: genre_encoder.transform([x])[0] if x in genre_encoder.classes_ else -1
),
    "duration": df["duration_num"]
})


df["Cluster"] = clustering.predict(x)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Netflix Filters")

country_list = sorted(
    set(
        country.strip()
        for row in df["country"].dropna().astype(str)
        for country in row.split(",")
    )
)

countries = st.sidebar.multiselect(
    "Country",
    country_list,
)

genre_list = sorted(
    set(
        genre.strip()
        for row in df["listed_in"].dropna().astype(str)
        for genre in row.split(",")
    )
)

genres = st.sidebar.multiselect(
    "Genre",
    genre_list,
)

types = st.sidebar.multiselect(
    "Content Type",
    sorted(df["type"].unique())
)

filtered_df = df.copy()

if countries:
    filtered_df = filtered_df[
        filtered_df["country"].apply(
            lambda x: any(c.strip() in countries for c in str(x).split(","))
        )
    ]

if genres:
    filtered_df = filtered_df[
        filtered_df["listed_in"].apply(
            lambda x: any(g.strip() in genres for g in str(x).split(","))
        )
    ]

if types:
    filtered_df = filtered_df[
        filtered_df["type"].isin(types)
    ]


st.title("🎬 Netflix Content Strategy Dashboard")
st.markdown("### Milestone 4 - Interactive Dashboard")

# Small text
st.caption(f"Filtered Rows: {filtered_df.shape[0]}")

# KPI Cards
total_titles = len(filtered_df)
movies = len(filtered_df[filtered_df["type"] == "Movie"])
tv_shows = len(filtered_df[filtered_df["type"] == "TV Show"])
countries = filtered_df["country"].dropna().str.split(",").explode().nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📚 Total Titles", total_titles)

with col2:
    st.metric("🎬 Movies", movies)

with col3:
    st.metric("📺 TV Shows", tv_shows)

with col4:
    st.metric("🌍 Countries", countries)

# OR a small metric box
# st.metric("Filtered Rows", filtered_df.shape[0])

tab1, tab2 = st.tabs(["Strategic Insights", "AI Analytics"])

# =====================================================
# TAB 1
# =====================================================
with tab1:

    st.subheader("Movies vs TV Shows")

    fig = px.histogram(filtered_df, x="type")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Countries")

top_country = filtered_df["country"].value_counts().head(10).reset_index()
top_country.columns = ["Country", "Titles"]

fig = px.bar(
    top_country,
    x="Country",
    y="Titles"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Genres")
top_genre = filtered_df["listed_in"].value_counts().head(10).reset_index()
top_genre.columns = ["Genre", "Titles"]

fig = px.bar(
    top_genre,
    x="Genre",
    y="Titles"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 2
# =====================================================
with tab2:

    st.subheader("Cluster Distribution")

    fig = px.histogram(filtered_df, x="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Explorer")

    cluster = st.selectbox(
        "Select Cluster",
        sorted(filtered_df["Cluster"].unique())
    )

    st.dataframe(filtered_df[filtered_df["Cluster"]==cluster])

    st.subheader("Predict Content Type")

    country = st.selectbox(
        "Country",
        sorted(df["country"].unique())
    )

    genre = st.selectbox(
        "Genre",
        sorted(df["listed_in"].unique())
    )

    duration = st.slider(
        "Duration",
        30,
        300,
        120
    )

    if st.button("Predict"):
   
        st.write("Button clicked")
        st.write("Country:", country)
        st.write("Genre:", genre)
        st.write("Duration:", duration)

    pred = classifier.predict([[
        country_encoder.transform([country])[0],
        genre_encoder.transform([genre])[0],
        duration
    ]])[0]

    st.write("Prediction:", pred)

    if pred == "Movie":
          st.success("Predicted Content Type: Movie")
    else:
          st.success("Predicted Content Type: TV Show")
    importance = pd.DataFrame({
        "Feature":["Country","Genre","Duration"],
        "Importance":classifier.feature_importances_
    })

    fig = px.bar(
        importance,
        x="Feature",
        y="Importance"
    )

    st.plotly_chart(fig, use_container_width=True)