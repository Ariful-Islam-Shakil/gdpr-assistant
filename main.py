from services.extract_pdf import ExtractPDF
from services.chunker import ChunkPDF
from services.vector_chunk import VectorChunk
from services.scrap_mapping import ScrapMapping
import os

def main():
    # Path to the GDPR PDF file
    pdf_path = "CELEX_32016R0679_EN_TXT.pdf"
    output_path = "output"
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    print(f"1. Extracting PDF to Markdown: {pdf_path}...")
    
    try:
        # Step 0: Scrap Mapping
        # scrap_mapping = ScrapMapping()
        # scrap_mapping.get_recitals_number()
        
        # Step 1: Extract PDF to Markdown
        extractor = ExtractPDF(pdf_path)
        md_file = os.path.join(output_path, "gdpr.md")
        md_file = extractor.save_markdown(md_file)
        print(f"   Successfully converted PDF to Markdown: {md_file}")
        
        # Step 2: Article Chunking
        print(f"2. Chunking Markdown into Articles: {md_file}...")
        chunker = ChunkPDF(md_file)
        article_json = os.path.join(output_path, "gdpr_articles.json")
        article_json = chunker.save_json(article_json)
        print(f"   Successfully chunked into Articles: {article_json}")

        # Step 2.5: Recitals Extraction
        print(f"2.5. Extracting Recitals: {md_file}...")
        recitals_json = os.path.join(output_path, "gdpr_recitals.json")
        recitals_json = chunker.save_recitals_json(recitals_json)
        
        # Step 3: Vector Chunking (Splitting articles into smaller parts)
        print(f"3. Splitting Articles into Vector Chunks (500 tokens): {article_json}...")
        vector_chunker = VectorChunk(article_json, max_tokens=500, overlap=50)
        vector_json = os.path.join(output_path, "gdpr_vector_chunks.json")
        vector_json = vector_chunker.save_json(vector_json)
        print(f"   Successfully created Vector Chunks: {vector_json}")
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
