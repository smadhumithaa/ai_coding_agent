import os
import json
import numpy as np
import google.generativeai as genai
import fnmatch
import time
from dotenv import load_dotenv

class SimpleRAG:
    def __init__(self, workspace_path):
        self.workspace_path = workspace_path
        self.index_file = os.path.join(workspace_path, '.ai_coder_index.json')
        self.documents = []  # List of dicts: {'path': str, 'chunk_id': int, 'content': str, 'embedding': list}
        
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY not found. RAG functionality will be disabled.")
        genai.configure(api_key=api_key)
        self.model_name = "models/gemini-embedding-2"

        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                print(f"Loaded {len(self.documents)} vectorized chunks from index.")
            except Exception as e:
                print(f"Failed to load RAG index: {e}")
                self.documents = []

    def save_index(self):
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f)
            print("Saved RAG index.")
        except Exception as e:
            print(f"Failed to save RAG index: {e}")

    def get_embedding(self, text):
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def get_query_embedding(self, text):
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def index_workspace(self, force=False):
        """Scan workspace and update embeddings for changed files."""
        if not os.getenv("GEMINI_API_KEY"):
            return "RAG is disabled (Missing GEMINI_API_KEY)"

        print("Indexing workspace...")
        ignore_dirs = {'.git', 'venv', 'node_modules', '__pycache__', 'dist', 'build'}
        ignore_exts = {'.pyc', '.exe', '.dll', '.so', '.png', '.jpg', '.jpeg', '.gif', '.mp4', '.zip', '.tar', '.gz'}
        
        # Keep track of current files
        current_files = set()
        new_docs = []

        for root, dirs, files in os.walk(self.workspace_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ignore_exts or file.startswith('.'):
                    continue
                    
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, self.workspace_path)
                current_files.add(rel_path)

                # Simple modified check (could be improved with mtime)
                # For simplicity in this demo, if force=True, we re-index everything.
                if not force and any(doc['path'] == rel_path for doc in self.documents):
                    continue

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    continue  # Skip binary files
                
                # Basic chunking: split by paragraphs or max 1000 chars
                # For a production app, use LangChain's RecursiveCharacterTextSplitter
                chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]
                
                for i, chunk in enumerate(chunks):
                    emb = self.get_embedding(f"File: {rel_path}\n\n{chunk}")
                    if emb:
                        new_docs.append({
                            'path': rel_path,
                            'chunk_id': i,
                            'content': chunk,
                            'embedding': emb
                        })
                        time.sleep(0.5) # Rate limit protection for free tier

        if force:
            self.documents = new_docs
        else:
            # Remove deleted files
            self.documents = [doc for doc in self.documents if doc['path'] in current_files]
            # Add new chunks
            self.documents.extend(new_docs)

        self.save_index()
        return f"Indexed {len(new_docs)} new chunks. Total chunks: {len(self.documents)}"

    def search(self, query, top_k=3):
        if not self.documents:
            return "Index is empty. Run workspace indexing first."
            
        query_emb = self.get_query_embedding(query)
        if not query_emb:
            return "Failed to generate query embedding."

        query_vec = np.array(query_emb)
        results = []

        for doc in self.documents:
            doc_vec = np.array(doc['embedding'])
            # Cosine similarity
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            results.append((similarity, doc))

        # Sort by highest similarity
        results.sort(key=lambda x: x[0], reverse=True)
        
        top_results = results[:top_k]
        
        formatted_results = []
        for sim, doc in top_results:
            formatted_results.append(f"--- File: {doc['path']} (Similarity: {sim:.2f}) ---\n{doc['content']}\n")
            
        return "\n\n".join(formatted_results)

if __name__ == "__main__":
    # Test script
    load_dotenv()
    rag = SimpleRAG(".")
    print(rag.index_workspace(force=True))
    print(rag.search("Where are the LLM providers configured?"))
