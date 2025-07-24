from sentence_transformers import SentenceTransformer
import os

def download_models():
    os.makedirs("models", exist_ok=True)
    
    fast_model = SentenceTransformer('all-MiniLM-L6-v2')
    fast_model.save("models/fast_model")
    
    precise_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    precise_model.save("models/precise_model")

def get_model_size(model_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return f"{total_size / (1024*1024):.1f} MB"

if __name__ == "__main__":
    download_models()