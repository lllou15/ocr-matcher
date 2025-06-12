import streamlit as st


class Settings:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    DB_HOST = st.secrets["DB_HOST"]
    DB_PORT = int(st.secrets["DB_PORT"])  # Ensure it's an integer
    DB_USER = st.secrets["DB_USER"]
    DB_NAME = st.secrets["DB_NAME"]
    DB_PASSWORD = st.secrets["DB_PASSWORD"]


settings = Settings()
