import os
import re
import csv
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
import streamlit as st

# Function to load YouTube API Key securely
def get_api_key():
    """
    Fetch API key from user input through the Streamlit interface.
    """
    return st.text_input("Enter YouTube API Key", type="password")

def get_video_id(url):
    """
    Extracts the video id from the given URL.
    """
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
    return video_id_match.group(1) if video_id_match else None

def get_video_title(video_id, api_key):
    """
    Fetches the title of the video using YouTube Data API.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videos().list(part='snippet', id=video_id)
    response = request.execute()
    return response['items'][0]['snippet']['title'] if response['items'] else 'Unknown Title'

def get_video_transcript(video_id):
    """
    Fetches the transcript of the video using youtube-transcript-api.
    """
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return []

def save_transcript_to_dataframe(transcript):
    """
    Converts the transcript into a DataFrame.
    """
    return pd.DataFrame([{'start': entry['start'], 'text': entry['text']} for entry in transcript])

def topic_modeling_nmf(transcript_df, n_topics=10, n_top_words=10):
    """
    Perform topic modeling using NMF and return topics and topic distribution.
    """
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(transcript_df['text'])
    nmf = NMF(n_components=n_topics, random_state=42).fit(tf)

    feature_names = tf_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(" ".join(topic_words))

    topic_distribution = nmf.transform(tf)
    return topics, topic_distribution

def identify_chapters(transcript_df, topic_distribution, threshold=60):
    """
    Identify logical breaks and consolidate into chapters.
    """
    transcript_df['dominant_topic'] = topic_distribution.argmax(axis=1)

    logical_breaks = [
        transcript_df['start'].iloc[i]
        for i in range(1, len(transcript_df))
        if transcript_df['dominant_topic'].iloc[i] != transcript_df['dominant_topic'].iloc[i - 1]
    ]

    consolidated_breaks = []
    last_break = None
    for break_point in logical_breaks:
        if last_break is None or break_point - last_break >= threshold:
            consolidated_breaks.append(break_point)
            last_break = break_point

    chapters = []
    for i, break_point in enumerate(consolidated_breaks):
        chapter_text = transcript_df[
            (transcript_df['start'] >= break_point) &
            (transcript_df['dominant_topic'] == transcript_df[transcript_df['start'] == break_point]['dominant_topic'].values[0])
        ]['text'].str.cat(sep=' ')

        vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
        tfidf_matrix = vectorizer.fit_transform([chapter_text])
        feature_names = vectorizer.get_feature_names_out()
        chapter_name = " ".join(feature_names)

        chapter_time = pd.to_datetime(break_point, unit='s').strftime('%H:%M:%S')
        chapters.append((chapter_time, f"Chapter {i + 1}: {chapter_name}"))

    return chapters

def main():
    st.title("YouTube Video Chaptering Tool")

    api_key = get_api_key()
    if not api_key:
        st.warning("Please enter your YouTube API Key to proceed.")
        return

    url = st.text_input("Enter YouTube Video URL")

    if url:
        video_id = get_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL. Please provide a valid one.")
            return

        st.write("Fetching video details...")
        title = get_video_title(video_id, api_key)

        st.write("Fetching transcript...")
        transcript = get_video_transcript(video_id)

        if not transcript:
            st.error("No transcript available for this video.")
            return

        st.success(f"Video Title: {title}")
        transcript_df = save_transcript_to_dataframe(transcript)

        st.write("Performing topic modeling...")
        topics, topic_distribution = topic_modeling_nmf(transcript_df)

        st.subheader("Identified Topics")
        for i, topic in enumerate(topics):
            st.write(f"Topic {i + 1}: {topic}")

        st.write("Identifying chapters...")
        chapters = identify_chapters(transcript_df, topic_distribution)

        st.subheader("Chapters")
        for time, name in chapters:
            st.write(f"{time} - {name}")

if __name__ == '__main__':
    main()
