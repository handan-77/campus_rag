import pdfplumber
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

def pdf_to_word(pdf_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(pdf_path)
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    docx_path = os.path.join(output_dir, f"{base_name}.docx")
    
    doc = Document()
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                for line in text.split('\n'):
                    p = doc.add_paragraph(line)
                    p.paragraph_format.space_after = Pt(2)
            
            if page_num < len(pdf.pages) - 1:
                doc.add_page_break()
    
    doc.save(docx_path)
    return docx_path

if __name__ == "__main__":
    pdf_path = r"C:\Users\30523\Desktop\关于做好2026—2027学年度院团学组织副部选拔的通知(1).pdf"
    result = pdf_to_word(pdf_path)
    print(f"转换完成: {result}")
