import streamlit as st


class Settings:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    DB_HOST = st.secrets["DB_HOST"]
    DB_NAME = st.secrets["DB_NAME"]
    DB_USER = st.secrets["DB_USER"]
    DB_PASSWORD = st.secrets["DB_PASSWORD"]


settings = Settings()
