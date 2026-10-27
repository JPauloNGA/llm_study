import requests
import time
import pandas as pd
import matplotlib.pyplot as plt

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_PS_URL = "http://localhost:11434/api/ps"

MODELOS = ["qwen2.5:1.5b", "llama3.2:1b", "gemma2:2b", "qwen2.5:3b"]

PROMPT_PT = "Explique em uma frase o que é computação de alta performance."
PROMPT_EN = "Explain in one sentence what high-performance computing is."

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

def rodar_teste(modelo, prompt, idioma):
    print(f"\n--- {modelo} | {idioma} ---")
    
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
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
    prompt_eval_count = data.get("prompt_eval_count", 0)
    tps = eval_count / eval_duration if eval_duration > 0 else 0
    
    # Mede VRAM logo após a geração (modelo ainda carregado)
    vram = get_vram(modelo)
    
    print(f"Tokens Prompt: {prompt_eval_count} | Tokens Gerados: {eval_count} | Tempo: {eval_duration:.2f}s | Velocidade: {tps:.2f} t/s | VRAM: {vram} GB")
    
    return {
        "Modelo": modelo,
        "Idioma": idioma,
        "Tokens Prompt": prompt_eval_count,
        "Tokens Gerados": eval_count,
        "Tempo (s)": round(eval_duration, 2),
        "Velocidade (t/s)": round(tps, 2),
        "VRAM (GB)": vram
    }

if __name__ == "__main__":
    resultados = []
    
    for modelo in MODELOS:
        warm_up(modelo)
        
        res_pt = rodar_teste(modelo, PROMPT_PT, "PT-BR")
        if res_pt: resultados.append(res_pt)
        
        res_en = rodar_teste(modelo, PROMPT_EN, "EN")
        if res_en: resultados.append(res_en)
    
    if resultados:
        df = pd.DataFrame(resultados)
        print("\n" + "="*60)
        print("RESUMO COMPARATIVO DE MODELOS")
        print("="*60)
        print(df.to_string(index=False))
        
        # Gráfico de velocidade por modelo e idioma
        fig, ax = plt.subplots(figsize=(10, 5))
        modelos_unicos = df['Modelo'].unique()
        x = range(len(modelos_unicos))
        width = 0.35
        
        pt_vals = [df[(df['Modelo'] == m) & (df['Idioma'] == 'PT-BR')]['Velocidade (t/s)'].values[0] if not df[(df['Modelo'] == m) & (df['Idioma'] == 'PT-BR')].empty else 0 for m in modelos_unicos]
        en_vals = [df[(df['Modelo'] == m) & (df['Idioma'] == 'EN')]['Velocidade (t/s)'].values[0] if not df[(df['Modelo'] == m) & (df['Idioma'] == 'EN')].empty else 0 for m in modelos_unicos]
        
        ax.bar([i - width/2 for i in x], pt_vals, width, label='PT-BR', color='green', alpha=0.8)
        ax.bar([i + width/2 for i in x], en_vals, width, label='EN', color='blue', alpha=0.8)
        
        ax.set_xlabel('Modelo')
        ax.set_ylabel('Velocidade (tokens/s)')
        ax.set_title('Comparação de Modelos: Qwen 1.5B, Llama 1B, Gemma 2B, Qwen 3B (RX 580)')
        ax.set_xticks(x)
        ax.set_xticklabels(modelos_unicos, rotation=15)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('../results/exp_modelos_v2.png')
        print("\nGráfico salvo: results/exp_modelos_v2.png")
        plt.show()
        
        # Salvar respostas completas
        with open('../results/exp_modelos_v2_respostas.txt', 'w', encoding='utf-8') as f:
            for r in resultados:
                f.write(f"=== {r['Modelo']} | {r['Idioma']} ===\n")
                f.write(f"Tempo: {r['Tempo (s)']}s | Velocidade: {r['Velocidade (t/s)']} t/s\n\n")
        print("Respostas salvas em: results/exp_modelos_v2_respostas.txt")