import streamlit as st
import requests
import tempfile

st.set_page_config(page_title="OCR Medical Bill Matcher", layout="centered")
st.title("OCR Medical Bill Matcher")

uploaded_file = st.file_uploader(
    "Upload your medical bill image", type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    st.image(file_path, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Match Items"):
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post("http://127.0.0.1:8007/upload", files=files)

        if response.status_code == 200:
            result = response.json()
            matched_items = result.get("matched", [])
            points = result.get("points", 0)

            st.success(
                f"Matched {len(matched_items)} item(s). Points earned: {points:.2f}"
            )
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

        else:
            st.error("Failed to process the image.")
