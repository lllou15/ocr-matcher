from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import pytesseract
import google.generativeai as genai
import os, io, json, re
from backend.rds_connection import RelationalDatabaseConnector
from backend.config import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")
rds = RelationalDatabaseConnector()


class Item(BaseModel):
    name: str
    manufacturer: str
    HSN: str
    pack: str
    batch: str
    exp_dt: str
    MRP: float
    qty: int
    rate: float
    PD_percent: float | None
    BD_percent: float | None
    value: float | None


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


@app.post("/upload")
async def process_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
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

        if best_score > 0.6:  # You can adjust this threshold as needed
            found_items.append(item)

    total_value = sum(i.get("value", 0) or 0 for i in found_items)
    points = total_value / 100

    return JSONResponse({"matched": found_items, "points": points})
