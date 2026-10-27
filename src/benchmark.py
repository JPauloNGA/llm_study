import requests
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELOS = ["qwen2.5:1.5b", "llama3.2:1b", "qwen2.5:3b"]

PROMPT_PT = "Explique em uma frase o que é computação de alta performance."
PROMPT_EN = "Explain in one sentence what high-performance computing is."

def warm_up(modelo):
    """Faz uma requisição rápida só para carregar o modelo na VRAM."""
    print(f"  [Warm-up] Carregando {modelo} na VRAM...")
    payload = {
        "model": modelo,
        "prompt": "OK",
        "stream": False,
        "options": {"num_predict": 1}
    }
    try:
        requests.post(OLLAMA_URL, json=payload, timeout=120)
    except Exception as e:
        print(f"  Erro no warm-up: {e}")

def rodar_teste(modelo, prompt, idioma):
    print(f"\n--- Testando: {modelo} | {idioma} ---")
    
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 50}
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.exceptions.Timeout:
        print("Timeout na requisição.")
        return None
    
    if response.status_code != 200:
        print(f"Erro na requisição: {response.status_code}")
        return None
    
    data = response.json()
    
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 1)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    
    # Usa APENAS o tempo de geração (eval_duration), não o total
    tempo_geracao_s = eval_duration_ns / 1e9
    tps = eval_count / tempo_geracao_s if tempo_geracao_s > 0 else 0
    
    print(f"Tokens Prompt: {prompt_eval_count} | Tokens Gerados: {eval_count} | Tempo Geração: {tempo_geracao_s:.2f}s | Velocidade: {tps:.2f} t/s")
    
    return {
        "Modelo": modelo,
        "Idioma": idioma,
        "Tokens Prompt": prompt_eval_count,
        "Tokens Gerados": eval_count,
        "Tempo Geração (s)": round(tempo_geracao_s, 2),
        "Velocidade (tokens/s)": round(tps, 2)
    }

if __name__ == "__main__":
    resultados = []
    
    for modelo in MODELOS:
        # Warm-up: carrega o modelo antes de medir
        warm_up(modelo)
        
        res_pt = rodar_teste(modelo, PROMPT_PT, "PT-BR")
        if res_pt: resultados.append(res_pt)
        
        res_en = rodar_teste(modelo, PROMPT_EN, "EN")
        if res_en: resultados.append(res_en)
    
    if resultados:
        df = pd.DataFrame(resultados)
        print("\n--- Tabela Final ---")
        print(df.to_string(index=False))
        
        # Gráfico de barras agrupadas
        fig, ax = plt.subplots(figsize=(10, 5))
        
        modelos_unicos = df['Modelo'].unique()
        x = range(len(modelos_unicos))
        width = 0.35
        
        pt_vals = [df[(df['Modelo'] == m) & (df['Idioma'] == 'PT-BR')]['Velocidade (tokens/s)'].values[0] if not df[(df['Modelo'] == m) & (df['Idioma'] == 'PT-BR')].empty else 0 for m in modelos_unicos]
        en_vals = [df[(df['Modelo'] == m) & (df['Idioma'] == 'EN')]['Velocidade (tokens/s)'].values[0] if not df[(df['Modelo'] == m) & (df['Idioma'] == 'EN')].empty else 0 for m in modelos_unicos]
        
        ax.bar([i - width/2 for i in x], pt_vals, width, label='PT-BR', color='green', alpha=0.8)
        ax.bar([i + width/2 for i in x], en_vals, width, label='EN', color='blue', alpha=0.8)
        
        ax.set_xlabel('Modelo')
        ax.set_ylabel('Velocidade (tokens/s)')
        ax.set_title('Comparação de Modelos: Velocidade PT-BR vs EN (RX 580 / Vulkan)')
        ax.set_xticks(x)
        ax.set_xticklabels(modelos_unicos)
        ax.legend()
        
        plt.tight_layout()
        
        if not os.path.exists('../results'):
            os.makedirs('../results')
        plt.savefig('../results/benchmark_modelos.png')
        print("\nGráfico salvo em: results/benchmark_modelos.png")
        plt.show()