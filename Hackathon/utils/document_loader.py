# utils/document_loader.py
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from langchain.schema import Document
import tempfile

def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    doc = fitz.open(tmp_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if len(text.strip()) < 50:
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(image, config="--psm 6")

        if text.strip():
            pages.append(Document(page_content=text, metadata={"page": i + 1}))

    doc.close()
    return pages
