import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using pypdf.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from all pages.
    """
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
    return text

def process_pdf(pdf_path):
    """
    Process a PDF file: extract text and save to TXT.

    Args:
        pdf_path (str): Path to the PDF file.
    """
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} does not exist.")
        return

    extracted_text = extract_text_from_pdf(pdf_path)

    # Save to a text file (you can adjust naming as needed)
    txt_path = os.path.splitext(pdf_path)[0] + "pdf.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

    print(f"Extracted text saved to {txt_path}")

if __name__ == "__main__":
    process_pdf("./RAG/retrieval/txt_extract/人事管理流程.pdf")