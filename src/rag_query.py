import requests
import json
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

OLLAMA_URL_GENERATE = "http://localhost:11434/api/generate"
OLLAMA_URL_EMBEDDINGS = "http://localhost:11434/api/embeddings"
MODELO_EMBEDDING = "nomic-embed-text"
MODELO_GERACAO = "jarvis"
INDEX_FILE = "./results/rag_index.json"

def carregar_indice():
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def gerar_embedding(texto):
    payload = {"model": MODELO_EMBEDDING, "prompt": texto}
    response = requests.post(OLLAMA_URL_EMBEDDINGS, json=payload, timeout=60)
    return response.json().get("embedding")

def similaridade_cosseno(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def buscar_trechos_relevantes(pergunta, indice, top_k=2):
    embedding_pergunta = gerar_embedding(pergunta)
    if not embedding_pergunta:
        return []
    similaridades = []
    for item in indice:
        sim = similaridade_cosseno(embedding_pergunta, item["embedding"])
        similaridades.append((sim, item))
    similaridades.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in similaridades[:top_k]]

def gerar_resposta(prompt, label):
    """Chama o Jarvis e retorna métricas."""
    payload = {
        "model": MODELO_GERACAO,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 150}
    }
    
    start = time.time()
    response = requests.post(OLLAMA_URL_GENERATE, json=payload, timeout=120)
    end = time.time()
    
    data = response.json()
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 1) / 1e9
    tps = eval_count / eval_duration if eval_duration > 0 else 0
    tempo_total = end - start
    
    print(f"  [{label}] Tempo: {tempo_total:.2f}s | Tokens: {eval_count} | Velocidade: {tps:.2f} t/s")
    print(f"  [{label}] Resposta: {data.get('response', '')[:200]}...")
    
    return {
        "tempo_total": tempo_total,
        "tokens": eval_count,
        "tps": tps,
        "resposta": data.get("response", "")
    }

def comparar_rag(pergunta):
    print(f"\n{'='*60}")
    print(f"Pergunta: {pergunta}")
    print(f"{'='*60}")
    
    # --- SEM RAG ---
    print("\n[SEM RAG]")
    prompt_sem_rag = pergunta
    res_sem = gerar_resposta(prompt_sem_rag, "Sem RAG")
    
    # --- COM RAG ---
    print("\n[COM RAG]")
    indice = carregar_indice()
    
    start_busca = time.time()
    trechos = buscar_trechos_relevantes(pergunta, indice, top_k=2)
    tempo_busca = time.time() - start_busca
    
    if trechos:
        print(f"  Busca: {tempo_busca:.2f}s | {len(trechos)} trechos encontrados")
        for t in trechos:
            print(f"    [{t['arquivo']}] {t['texto'][:80]}...")
        contexto = "\n\n".join([t["texto"] for t in trechos])
    else:
        print("  Nenhum trecho encontrado.")
        contexto = ""
    
    prompt_com_rag = f"""Use o contexto abaixo para responder à pergunta. Se o contexto não contiver a resposta, diga que não sabe.

Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:"""
    
    res_com = gerar_resposta(prompt_com_rag, "Com RAG")
    
    return {
        "pergunta": pergunta,
        "sem_rag_tempo": res_sem["tempo_total"],
        "sem_rag_tps": res_sem["tps"],
        "sem_rag_resposta": res_sem["resposta"],
        "com_rag_tempo_busca": tempo_busca,
        "com_rag_tempo_geracao": res_com["tempo_total"],
        "com_rag_tempo_total": tempo_busca + res_com["tempo_total"],
        "com_rag_tps": res_com["tps"],
        "com_rag_resposta": res_com["resposta"]
    }

if __name__ == "__main__":
    perguntas = [
        "Qual a diferença entre microcontrolador e microprocessador?",
        "O que é um servo motor?",
    ]
    
    resultados = []
    for pergunta in perguntas:
        res = comparar_rag(pergunta)
        resultados.append(res)
    
    # Tabela comparativa
    print("\n" + "="*60)
    print("RESUMO COMPARATIVO: COM RAG vs. SEM RAG")
    print("="*60)
    
    df = pd.DataFrame([{
        "Pergunta": r["pergunta"][:40] + "...",
        "Sem RAG (s)": round(r["sem_rag_tempo"], 2),
        "Sem RAG (t/s)": round(r["sem_rag_tps"], 2),
        "Com RAG (busca)": round(r["com_rag_tempo_busca"], 2),
        "Com RAG (geração)": round(r["com_rag_tempo_geracao"], 2),
        "Com RAG (total)": round(r["com_rag_tempo_total"], 2),
        "Com RAG (t/s)": round(r["com_rag_tps"], 2),
    } for r in resultados])
    
    print(df.to_string(index=False))
    
    # Gráfico comparativo de tempo
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(resultados))
    width = 0.35
    
    sem_rag_tempos = [r["sem_rag_tempo"] for r in resultados]
    com_rag_tempos = [r["com_rag_tempo_total"] for r in resultados]
    
    ax.bar([i - width/2 for i in x], sem_rag_tempos, width, label='Sem RAG', color='#e74c3c', alpha=0.8)
    ax.bar([i + width/2 for i in x], com_rag_tempos, width, label='Com RAG (busca + geração)', color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Pergunta')
    ax.set_ylabel('Tempo Total (s)')
    ax.set_title('Experimento 10: Impacto do RAG no Tempo de Resposta (RX 580)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}" for i in x])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('../results/exp10_rag_tempo.png')
    print("\nGráfico salvo: results/exp10_rag_tempo.png")
    plt.show()
    
    # Salvar respostas completas para análise manual
    with open('../results/exp10_respostas.txt', 'w', encoding='utf-8') as f:
        for r in resultados:
            f.write(f"=== {r['pergunta']} ===\n")
            f.write(f"\n--- SEM RAG ---\n{r['sem_rag_resposta']}\n")
            f.write(f"\n--- COM RAG ---\n{r['com_rag_resposta']}\n\n")
    print("Respostas completas salvas em: results/exp10_respostas.txt")