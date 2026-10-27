import requests
import json
import os
import glob

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODELO_EMBEDDING = "nomic-embed-text"
DOCS_DIR = "./docs"
INDEX_FILE = "./results/rag_index.json"

def gerar_embedding(texto):
    """Gera o embedding de um texto usando o nomic-embed-text."""
    payload = {
        "model": MODELO_EMBEDDING,
        "prompt": texto
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    if response.status_code != 200:
        print(f"Erro ao gerar embedding: {response.status_code}")
        return None
    return response.json().get("embedding")

def indexar_documentos():
    """Lê todos os arquivos .txt da pasta docs e gera embeddings."""
    if not os.path.exists(DOCS_DIR):
        print(f"Pasta {DOCS_DIR} não encontrada. Crie a pasta e adicione arquivos .txt.")
        return
    
    arquivos = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    if not arquivos:
        print("Nenhum arquivo .txt encontrado em docs/.")
        return
    
    print(f"Encontrados {len(arquivos)} arquivos. Gerando embeddings...")
    
    indice = []
    for arquivo in arquivos:
        with open(arquivo, "r", encoding="utf-8") as f:
            texto = f.read().strip()
        
        if not texto:
            continue
        
        # Divide o texto em chunks de ~500 caracteres (para melhor granularidade)
        chunks = [texto[i:i+500] for i in range(0, len(texto), 500)]
        
        for i, chunk in enumerate(chunks):
            print(f"  Processando {os.path.basename(arquivo)} - chunk {i+1}/{len(chunks)}...")
            embedding = gerar_embedding(chunk)
            if embedding:
                indice.append({
                    "arquivo": os.path.basename(arquivo),
                    "chunk_id": i,
                    "texto": chunk,
                    "embedding": embedding
                })
    
    # Salva o índice
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(indice, f, ensure_ascii=False)
    
    print(f"\nÍndice salvo em {INDEX_FILE} com {len(indice)} chunks.")

if __name__ == "__main__":
    indexar_documentos()