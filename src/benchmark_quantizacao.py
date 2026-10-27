import requests
import time
import pandas as pd
import matplotlib.pyplot as plt

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_PS_URL = "http://localhost:11434/api/ps"

# Q4 é a tag padrão do Qwen 2.5 1.5B no Ollama
# Q8 é a versão de 8 bits que você já baixou
VARIANTES = [
    ("qwen2.5:1.5b", "Q4_K_M (4-bit)"),
    ("qwen2.5:1.5b-instruct-q8_0", "Q8_0 (8-bit)")
]

PROMPT = "Explique em uma frase o que é computação de alta performance."

def warm_up(modelo):
    print(f"  [Warm-up] Carregando {modelo}...")
    payload = {"model": modelo, "prompt": "OK", "stream": False, "options": {"num_predict": 1}}
    try:
        requests.post(OLLAMA_URL, json=payload, timeout=120)
    except Exception as e:
        print(f"  Erro no warm-up: {e}")

def get_vram(modelo_nome):
    """Lê a VRAM usada pelo modelo via /api/ps."""
    try:
        resp = requests.get(OLLAMA_PS_URL, timeout=10)
        for m in resp.json().get("models", []):
            if modelo_nome in m["name"]:
                return round(m["size_vram"] / (1024**3), 2)  # GB
    except Exception:
        pass
    return None

def rodar_teste(modelo_tag, label):
    print(f"\n--- Testando: {label} ({modelo_tag}) ---")
    warm_up(modelo_tag)
    
    payload = {
        "model": modelo_tag,
        "prompt": PROMPT,
        "stream": False,
        "keep_alive": "5m",  # Mantém na VRAM para medir depois
        "options": {"num_predict": 50}
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.exceptions.Timeout:
        print("Timeout.")
        return None
    
    if response.status_code != 200:
        print(f"Erro: {response.status_code}")
        return None
    
    data = response.json()
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 1) / 1e9
    tps = eval_count / eval_duration if eval_duration > 0 else 0
    vram = get_vram(modelo_tag)
    
    print(f"Tokens Gerados: {eval_count} | Tempo: {eval_duration:.2f}s | Velocidade: {tps:.2f} t/s | VRAM: {vram} GB")
    print(f"Resposta: {data.get('response', '')[:100]}...")
    
    return {
        "Quantização": label,
        "Tag": modelo_tag,
        "Tokens Gerados": eval_count,
        "Tempo (s)": round(eval_duration, 2),
        "Velocidade (t/s)": round(tps, 2),
        "VRAM (GB)": vram
    }

if __name__ == "__main__":
    resultados = []
    for tag, label in VARIANTES:
        res = rodar_teste(tag, label)
        if res:
            resultados.append(res)
    
    if resultados:
        df = pd.DataFrame(resultados)
        print("\n" + "="*60)
        print("RESUMO: Q4 vs Q8")
        print("="*60)
        print(df.to_string(index=False))
        
        # Gráfico comparativo
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        # Barras de velocidade
        colors = ['#3498db', '#e67e22']
        bars = ax1.bar(df['Quantização'], df['Velocidade (t/s)'], color=colors, alpha=0.8, width=0.5)
        ax1.set_ylabel('Velocidade (tokens/s)', color='black')
        ax1.set_title('Experimento 12: Impacto da Quantização (Q4 vs Q8) - Qwen 1.5B')
        ax1.set_ylim(0, max(df['Velocidade (t/s)']) * 1.4)
        
        # Rótulos nas barras
        for i, v in enumerate(df['Velocidade (t/s)']):
            ax1.text(i, v + 0.5, f'{v:.2f} t/s', ha='center', fontweight='bold')
        
        # Linha de VRAM no eixo secundário
        ax2 = ax1.twinx()
        ax2.plot(df['Quantização'], df['VRAM (GB)'], color='red', marker='o', linewidth=2, label='VRAM (GB)')
        ax2.set_ylabel('VRAM (GB)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Rótulos de VRAM
        for i, v in enumerate(df['VRAM (GB)']):
            ax2.text(i, v + 0.05, f'{v:.2f} GB', ha='center', color='red', fontweight='bold')
        
        fig.tight_layout()
        plt.savefig('../results/exp12_quantizacao.png')
        print("\nGráfico salvo: results/exp12_quantizacao.png")
        plt.show()