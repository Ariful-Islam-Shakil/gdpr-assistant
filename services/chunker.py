import json
import re
import os

class ChunkPDF:
    """
    A class to chunk Markdown content from GDPR into articles with associated metadata.
    """
    def __init__(self, md_path: str):
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Markdown file not found at: {md_path}")
        self.md_path = md_path
        self.recital_map = self.load_recital_map()

    def load_recital_map(self):
        map_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "scrap_mapping.json")
        with open(map_path, "r") as f:
            return json.load(f)
    def get_recital_by_article(self, article):
        for item in self.recital_map:
            if str(item["article"]) == str(article):
                return item["recitals"]
        return []

    def extract_recitals(self):
        """
        Parses the Markdown file and splits it into recitals (1-173).
        Each recital is a dictionary with number and content.
        """
        with open(self.md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # We only care about the content before CHAPTER I
        pre_articles_content = content.split('**CHAPTER I**')[0]
        
        # Pattern to match (1), (2), etc. at the start of a line
        recital_pattern = re.compile(r'^\((\d+)\)\s+(.*)', re.MULTILINE)
        
        recitals = []
        matches = list(recital_pattern.finditer(pre_articles_content))
        
        for i in range(len(matches)):
            start_pos = matches[i].start()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(pre_articles_content)
            
            raw_content = pre_articles_content[start_pos:end_pos].strip()
            # Remove the (N) prefix from the content
            recital_text = re.sub(r'^\(\d+\)\s+', '', raw_content)
            # Remove any page headers or noise that might have been caught between recitals
            recital_text = re.sub(r'\d+\.\d+\.\d+\s+L\s+\d+\/\d+\s+Official\s+Journal\s+of\s+the\s+European\s+Union\s+EN', '', recital_text).strip()

            recitals.append({
                "number": matches[i].group(1),
                "content": recital_text
            })

        return recitals

    def save_recitals_json(self, output_path: str = None):
        """
        Saves the recitals to a JSON file.
        """
        if not output_path:
            output_path = os.path.join(os.path.dirname(self.md_path), "gdpr_recitals.json")
            
        recitals = self.extract_recitals()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recitals, f, indent=4, ensure_ascii=False)
            
        print(f"   Successfully extracted {len(recitals)} recitals to: {output_path}")
        return output_path
     
    def chunk(self):
        """
        Parses the Markdown file and splits it into chunks based on Articles.
        Each chunk is a dictionary with content and metadata (Chapter, Article, Chapter Name, Article Name).
        """
        with open(self.md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Regex patterns
        chapter_pattern = re.compile(r'\*\*CHAPTER\s+([IVXLCDM]+)\*\*', re.IGNORECASE)
        article_pattern = re.compile(r'\*\*Article\s+(\d+)\*\*', re.IGNORECASE)

        chunks = []
        current_chapter = "Unknown"
        current_chapter_name = "Unknown"
        current_article = "Unknown"
        current_article_name = "Unknown"
        
        waiting_for_chapter_name = False
        waiting_for_article_name = False
        
        # Split by lines to iterate
        lines = content.split('\n')
        
        current_chunk_lines = []
        
        for line in lines:
            line_strip = line.strip()
            
            # Check for Chapter
            chapter_match = chapter_pattern.search(line_strip)
            if chapter_match:
                # Save previous chunk if exists before changing chapter context
                if current_article != "Unknown" and current_chunk_lines:
                    chunks.append({
                        "content": "\n".join(current_chunk_lines).strip(),
                        "metadata": {
                            "chapter": current_chapter,
                            "chapter_name": current_chapter_name,
                            "article": current_article,
                            "article_name": current_article_name
                        }
                    })
                    current_chunk_lines = []
                    current_article = "Unknown"
                    current_article_name = "Unknown"

                current_chapter = chapter_match.group(0).replace('*', '')
                waiting_for_chapter_name = True
                continue
                
            # Check for Article
            article_match = article_pattern.search(line_strip)
            if article_match:
                # Save previous chunk if exists
                if current_article != "Unknown" and current_chunk_lines:
                    chunks.append({
                        "content": "\n".join(current_chunk_lines).strip(),
                        "metadata": {
                            "chapter": current_chapter,
                            "chapter_name": current_chapter_name,
                            "article": current_article,
                            "article_name": current_article_name
                        }
                    })
                    current_chunk_lines = []
                
                current_article = article_match.group(1)
                current_article_name = "Unknown"
                waiting_for_article_name = True
                current_chunk_lines.append(line_strip)
                continue
            
            # Capture names if waiting
            if line_strip:
                if waiting_for_chapter_name:
                    current_chapter_name = line_strip.strip('*').strip()
                    waiting_for_chapter_name = False
                elif waiting_for_article_name:
                    current_article_name = line_strip.strip('*').strip()
                    waiting_for_article_name = False

            # Append line to current chunk
            if current_article != "Unknown":
                current_chunk_lines.append(line)

        # Append last chunk
        if current_article != "Unknown" and current_chunk_lines:
            chunks.append({
                "content": "\n".join(current_chunk_lines).strip(),
                "metadata": {
                    "chapter": current_chapter,
                    "chapter_name": current_chapter_name,
                    "article": current_article,
                    "article_name": current_article_name
                }
            })

        for chunk in chunks:
            chunk["metadata"]["recitals"] = self.get_recital_by_article(chunk["metadata"]["article"])
        return chunks

    def save_json(self, output_path: str = None):
        """
        Saves the chunks to a JSON file.
        """
        if not output_path:
            output_path = self.md_path.rsplit('.', 1)[0] + "_chunks.json"
            
        chunks = self.chunk()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
            
        return output_path 