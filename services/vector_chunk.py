import json
import os
import tiktoken

class VectorChunk:
    def __init__(self, 
                 chunk_path: str = 'CELEX_32016R0679_EN_TXT_chunks.json', 
                 max_tokens: int = 500, 
                 overlap: int = 50,
                 model_name: str = "gpt-4"):
        self.chunk_path = chunk_path
        self.max_tokens = max_tokens
        self.overlap = overlap
        # Initialize the tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def split_text(self, text: str) -> list:
        """
        Splits text into chunks based on token count with given overlap.
        """
        # Encode the text into tokens
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.max_tokens:
            return [text]
            
        chunks = []
        start = 0
        while start < len(tokens):
            # Calculate end based on max_tokens
            end = start + self.max_tokens
            
            # Get the slice of tokens
            chunk_tokens = tokens[start:end]
            
            # Decode tokens back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text.strip())
            
            if end >= len(tokens):
                break
                
            # Move start forward: current end minus overlap
            start = end - self.overlap
            
            # Safety check to avoid infinite loops
            if self.overlap >= self.max_tokens:
                start = end # No overlap if overlap is too large
                
        return chunks

    def process(self):
        """
        Reads the initial chunks JSON and splits them into further sub-chunks (vector chunks).
        """
        if not os.path.exists(self.chunk_path):
            raise FileNotFoundError(f"Chunk file not found: {self.chunk_path}")

        with open(self.chunk_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vector_chunks = []

        for item in data:
            content = item.get('content', '')
            # header = item.get('header', '')
            metadata = item.get('metadata', {})

            sub_chunks = self.split_text(content)

            for i, sub_content in enumerate(sub_chunks):
                # Create a new metadata copy to avoid modifying the original
                new_metadata = metadata.copy()
                new_metadata['part'] = f"part {i + 1}"
                # Add token count to metadata for verification
                new_metadata['token_count'] = len(self.encoding.encode(sub_content))
                
                vector_chunks.append({
                    "content": sub_content,
                    "metadata": new_metadata
                })

        return vector_chunks

    def save_json(self, output_path: str = None):
        """
        Saves the processed vector chunks to a JSON file.
        """
        if not output_path:
            output_path = self.chunk_path.rsplit('.', 1)[0] + "_vector.json"
            
        vector_chunks = self.process()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vector_chunks, f, indent=4, ensure_ascii=False)
            
        return output_path

if __name__ == "__main__":
    # Test the class with token-based settings
    chunker = VectorChunk(max_tokens=500, overlap=50)
    saved_path = chunker.save_json()
    print(f"Token-based vector chunks saved to: {saved_path}")