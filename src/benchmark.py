import requests
import json
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO = "qwen2.5:1.5b"

PROMPT_PT = "Explique em uma frase o que é computação de alta performance."
PROMPT_EN = "Explain in one sentence what high-performance computing is."

def rodar_teste(prompt, idioma):
    print(f"\n--- Testando: {idioma} ---")
    
    payload = {
        "model": MODELO,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 50} # Limite de tokens gerados
    }
    
    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload)
    end = time.time()
    
    if response.status_code != 200:
        print(f"Erro na requisição: {response.status_code}")
        return None
    
    data = response.json()
    
    # O Ollama retorna métricas em nanosegundos
    eval_count = data.get("eval_count", 0) # Tokens gerados
    eval_duration_ns = data.get("eval_duration", 1) # Tempo de geração
    prompt_eval_count = data.get("prompt_eval_count", 0) # Tokens do prompt
    total_duration_ns = data.get("total_duration", 1)
    
    tps = eval_count / (eval_duration_ns / 1e9) # Tokens por segundo
    tempo_total_s = total_duration_ns / 1e9
    
    print(f"Tokens do Prompt: {prompt_eval_count}")
    print(f"Tokens Gerados: {eval_count}")
    print(f"Tempo Total: {tempo_total_s:.2f}s")
    print(f"Velocidade: {tps:.2f} tokens/segundo")
    print(f"Resposta: {data['response'][:80]}...")
    
    return {
        "Idioma": idioma,
        "Tokens Prompt": prompt_eval_count,
        "Tokens Gerados": eval_count,
        "Tempo (s)": round(tempo_total_s, 2),
        "Velocidade (tokens/s)": round(tps, 2)
    }

if __name__ == "__main__":
    resultados = []
    
    res_pt = rodar_teste(PROMPT_PT, "PT-BR")
    if res_pt: resultados.append(res_pt)
    
    res_en = rodar_teste(PROMPT_EN, "EN")
    if res_en: resultados.append(res_en)
    
    if resultados:
        df = pd.DataFrame(resultados)
        print("\n--- Tabela Final ---")
        print(df.to_string(index=False))
        
        # Gráfico Comparativo
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        # Barras de Velocidade
        ax1.bar(df['Idioma'], df['Velocidade (tokens/s)'], color=['green', 'blue'], alpha=0.7, width=0.4)
        ax1.set_ylabel('Velocidade (tokens/s)', color='black')
        ax1.set_title('Benchmark: Prompt PT-BR vs Inglês (Ollama + Vulkan / RX 580)')
        
        # Linha de Tokens do Prompt (eixo secundário)
        ax2 = ax1.twinx()
        ax2.plot(df['Idioma'], df['Tokens Prompt'], color='red', marker='o', linewidth=2, label='Tokens do Prompt')
        ax2.set_ylabel('Qtd. Tokens do Prompt', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        fig.tight_layout()
        
        if not os.path.exists('../results'):
            os.makedirs('../results')
        plt.savefig('../results/benchmark_idiomas.png')
        print("\nGráfico salvo em: results/benchmark_idiomas.png")
        plt.show()