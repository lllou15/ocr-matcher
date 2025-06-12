import streamlit as st
import tempfile
from PIL import Image
import pytesseract
import google.generativeai as genai
import json, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.rds_connection import RelationalDatabaseConnector
from backend.config import settings

# Set up Streamlit UI
st.set_page_config(page_title="OCR Medical Bill Matcher", layout="centered")
st.title("OCR Medical Bill Matcher")

# Set up Gemini and DB
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")
rds = RelationalDatabaseConnector()


# Utility functions
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_similarity(text1, text2):
    texts = [normalize(text1), normalize(text2)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return sim[0][0]


def process_image_and_match_items(image):
    ocr_text = pytesseract.image_to_string(image)

    prompt = f"""
    You are a data extraction assistant. Extract JSON array of line items from medical bill OCR text:
    Output format strictly:
    [{{"manufacturer": "", "name": "", "HSN": "", "pack": "", "batch": "", "exp_dt": "", "MRP": 0.0, "qty": 0, "rate": 0.0, "PD%": null, "BD%": null, "value": 0.0}}]
    Input:
    {ocr_text}
    """

    response = model.generate_content(prompt)
    match = re.search(r"\[\s*{.*?}\s*]", response.text, re.DOTALL)

    try:
        parsed = json.loads(match.group(0)) if match else []
    except:
        parsed = []

    found_items = []
    query = "SELECT prod_name FROM medicines"
    candidates = rds.execute_fetch_query_on_rds(query)
    prod_names = [c[0] for c in candidates if c[0]]

    for item in parsed:
        name = item.get("name")
        if not name:
            continue

        similarities = [compute_similarity(name, candidate) for candidate in prod_names]
        best_idx = similarities.index(max(similarities))
        best_name = prod_names[best_idx]
        best_score = similarities[best_idx]

        if best_score > 0.6:
            found_items.append(item)

    total_value = sum(i.get("value", 0) or 0 for i in found_items)
    points = total_value / 100

    return found_items, points


# UI logic
uploaded_file = st.file_uploader(
    "Upload your medical bill image", type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    st.image(file_path, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Match Items"):
        with Image.open(file_path) as image:
            matched_items, points = process_image_and_match_items(image)

        st.success(f"Matched {len(matched_items)} item(s). Points earned: {points:.2f}")

        if matched_items:
            simplified_items = [
                {
                    "Manufacturer": item.get("manufacturer", ""),
                    "Name": item.get("name", ""),
                    "HSN": item.get("HSN", ""),
                    "Pack": item.get("pack", ""),
                }
                for item in matched_items
            ]
            st.dataframe(simplified_items, use_container_width=True)
        else:
            st.warning("No valid items found to display.")
