#                            **Video Chaptering Tool**

## This project provides a Streamlit-based application that allows users to:
  1. Extract and analyze transcripts from YouTube videos.
  2. Perform topic modeling to identify themes within the transcript.
  3. Automatically segment the video into chapters based on the content.

It leverages the YouTube Data API, YouTube Transcript API, and machine learning techniques like Non-negative Matrix Factorization (NMF) to create meaningful chapters.

## **Requirements**
    1.Python 3.x
    2.Google API Client Library
    3.YouTube API Key
    4.YouTube Data API
    5.Streamlit
    6.Pandas
    7.NumPy
    8.Scikit-learn
    9.NLTK
    10.youtube-transcript-api

## **Installation**
    1. Clone the repository: git clone https://github.com/Adi-ti09/video-transcript-analysis.git
    2. Install the required libraries: pip install -r requirements.txt
    3. Set up your YouTube API key: export API_KEY="YOUR_API_KEY"

## **Get your API**

   ### **Create a Google Developers Console project:**
        1. Go to the Google Developers Console website [Link Text](console.developers.google.com).
        2. Create a new project or select an existing one.
    
   ### **Enable the Youtube API**
        1. In the sidebar, click on "APIs & Services" > "Dashboard".
        2. Click "Enable APIs and Services" and search for "YouTube Data API".
        3. Click on "YouTube Data API" and click on the "Enable" button.

   ### **Create credentials for your project:**
        1. Click on "Navigation menu" (three horizontal lines in the top left corner) > "APIs & Services" > "Credentials".
        2. Click on "Create Credentials" > "OAuth client ID".
        3. Select "Other" as the application type.
        4. Enter a name for your client ID and authorized JavaScript origins.
        5. Click on the "Create" button.

   ### ** Get your API key**
        1. In the "APIs & Services" > "Credentials" page, find the "API key" section.
        2. Click on the "Create API key" button.
        3. Copy the API key and store it securely.

## **Usage**
    1. Run the script "streamlit run you_project_path video_transcript.py
    2. Enter you Youtube API.
    3. Enter the video ID or URL: https://www.youtube.com/watch?v=VIDEO_ID
    4. The script will retrieve the video transcript and perform analysis.

## **Key Functionalities**
    1. Transcript Extraction: Fetches subtitles/transcripts for YouTube videos, supporting multiple languages.
    2. Topic Modeling: Uses NMF to identify recurring themes in the video's content.
    3. Dynamic Chapter Generation: Automatically determines logical breaks in content and generates chapter titles using TF-IDF.

## **Limitations**
    1. Transcript availability depends on the video's settings (only works if subtitles are enabled).
    2. The accuracy of topic modeling may vary based on the video's language and content.

## **Contributing**
    Contributions are welcome! Please submit a pull request with your changes.

## **Acknowledgments**
    * This project uses the YouTube API and is subject to the YouTube API Terms of Service.
    * This project uses the NLTK library, which is licensed under the Apache License.
    * This project uses the scikit-learn library, which is licensed under the BSD License.



        



