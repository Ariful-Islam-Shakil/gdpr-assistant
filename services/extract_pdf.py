import fitz
import os

class ExtractPDF:
    """
    A class to extract content from PDF files and convert them to Markdown format.
    Uses PyMuPDF (fitz) for high-quality text and layout extraction.
    """
    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path
        self.doc = fitz.open(self.pdf_path)

    def to_markdown(self) -> str:
        """
        Converts the PDF content to a Markdown string.
        Attempts to preserve headers based on font size and formatting.
        """
        md_text = ""
        
        for page_num, page in enumerate(self.doc):
            # Extract blocks of text with structural info
            blocks = page.get_text("dict")["blocks"]
            
            # Sort blocks: top to bottom, then left to right
            blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            
            for b in blocks:
                if b["type"] == 0:  # Text block
                    block_text = ""
                    for line in b["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            content = span["text"] # Keep original spacing for spans
                            if not content.strip():
                                continue
                            
                            size = span["size"]
                            flags = span["flags"]  # 2 = bold, 4 = italic
                            
                            # Heuristic for headers
                            if size > 14:
                                line_text = f"\n# {content.strip()}\n"
                            elif size > 12:
                                line_text = f"\n## {content.strip()}\n"
                            elif flags & 2:  # Bold
                                line_text += f"**{content.strip()}**"
                            else:
                                line_text += content
                        
                        if line_text:
                            # Add space between lines if not a header
                            if "#" in line_text:
                                block_text += line_text + "\n"
                            else:
                                block_text += line_text.strip() + " "
                    
                    if block_text:
                        # Clean up multiple spaces
                        if "#" in block_text:
                            md_text += block_text.strip() + "\n\n"
                        else:
                            cleaned_text = " ".join(block_text.split())
                            md_text += cleaned_text + "\n\n"
        
        return md_text


    def save_markdown(self, output_path: str = None) -> str:
        """
        Saves the extracted Markdown content to a file.
        Returns the path to the saved file.
        """
        if not output_path:
            output_path = self.pdf_path.rsplit('.', 1)[0] + ".md"
            
        markdown_content = self.to_markdown()
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        return output_path

    def __del__(self):
        """Ensure the document is closed when the object is destroyed."""
        if hasattr(self, "doc"):
            self.doc.close()