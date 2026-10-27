import requests
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELOS = ["qwen2.5:1.5b", "llama3.2:1b", "gemma2:2b", "qwen2.5:3b"]
PROMPT_PT = "Explique em uma frase o que é computação de alta performance."
PROMPT_EN = "Explain in one sentence what high-performance computing is."

# Texto longo para o teste de KV Caching
CONTEXTO_LONGO = """
A computação de alta performance (HPC) é um campo da ciência da computação que se dedica 
ao desenvolvimento de sistemas e técnicas para processar grandes volumes de dados em alta velocidade. 
Ela é utilizada em diversas áreas, como simulações climáticas, análise genômica, inteligência artificial, 
renderização de imagens e cálculos financeiros complexos. Os sistemas de HPC geralmente envolvem 
clusters de computadores interconectados por redes de alta velocidade, além de GPUs especializadas 
para acelerar cálculos paralelos. A eficiência energética e o resfriamento desses sistemas são desafios 
importantes, pois o consumo de energia pode ser muito alto.
"""

PERGUNTA_RELACIONADA = "Resuma o texto acima em uma frase curta."

def warm_up(modelo):
    """Carrega o modelo na VRAM antes de medir."""
    print(f"  [Warm-up] Carregando {modelo}...")
    payload = {"model": modelo, "prompt": "OK", "stream": False, "options": {"num_predict": 1}}
    try:
        requests.post(OLLAMA_URL, json=payload, timeout=120)
    except Exception as e:
        print(f"  Erro no warm-up: {e}")

def chamar_ollama(modelo, prompt, contexto_extra=""):
    """Faz uma requisição e retorna as métricas."""
    prompt_completo = contexto_extra + "\n" + prompt if contexto_extra else prompt
    payload = {
        "model": modelo,
        "prompt": prompt_completo,
        "stream": False,
        "options": {"num_predict": 50}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.exceptions.Timeout:
        return None
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 1)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    tempo_geracao_s = eval_duration_ns / 1e9
    tps = eval_count / tempo_geracao_s if tempo_geracao_s > 0 else 0
    
    return {
        "tokens_prompt": prompt_eval_count,
        "tokens_gerados": eval_count,
        "tempo_geracao": tempo_geracao_s,
        "tps": tps,
        "resposta": data.get("response", "")
    }

# ============================================================
# EXPERIMENTO 1: PT-BR vs EN
# ============================================================
def experimento_idiomas(modelo="qwen2.5:1.5b"):
    print(f"\n{'='*50}")
    print(f"EXPERIMENTO 1: PT-BR vs EN ({modelo})")
    print(f"{'='*50}")
    
    warm_up(modelo)
    resultados = []
    
    for prompt, idioma in [(PROMPT_PT, "PT-BR"), (PROMPT_EN, "EN")]:
        print(f"\n--- {idioma} ---")
        res = chamar_ollama(modelo, prompt)
        if res:
            print(f"Tokens Prompt: {res['tokens_prompt']} | Tokens Gerados: {res['tokens_gerados']} | Tempo: {res['tempo_geracao']:.2f}s | Velocidade: {res['tps']:.2f} t/s")
            resultados.append({
                "Idioma": idioma,
                "Tokens Prompt": res['tokens_prompt'],
                "Tokens Gerados": res['tokens_gerados'],
                "Tempo (s)": round(res['tempo_geracao'], 2),
                "Velocidade (tokens/s)": round(res['tps'], 2)
            })
    
    if resultados:
        df = pd.DataFrame(resultados)
        print("\n--- Tabela ---")
        print(df.to_string(index=False))
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(df['Idioma'], df['Velocidade (tokens/s)'], color=['green', 'blue'], alpha=0.7, width=0.4)
        ax1.set_ylabel('Velocidade (tokens/s)')
        ax1.set_title(f'Experimento 1: PT-BR vs EN ({modelo})')
        
        ax2 = ax1.twinx()
        ax2.plot(df['Idioma'], df['Tokens Prompt'], color='red', marker='o', linewidth=2, label='Tokens do Prompt')
        ax2.set_ylabel('Tokens do Prompt', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        fig.tight_layout()
        plt.savefig('../results/exp1_idiomas.png')
        print("\nGráfico salvo: results/exp1_idiomas.png")
        plt.close()

# ============================================================
# EXPERIMENTO 2: Comparação de Modelos
# ============================================================
def experimento_modelos():
    print(f"\n{'='*50}")
    print("EXPERIMENTO 2: Comparação de Modelos")
    print(f"{'='*50}")
    
    resultados = []
    
    for modelo in MODELOS:
        warm_up(modelo)
        
        for prompt, idioma in [(PROMPT_PT, "PT-BR"), (PROMPT_EN, "EN")]:
            print(f"\n--- {modelo} | {idioma} ---")
            res = chamar_ollama(modelo, prompt)
            if res:
                print(f"Tokens: {res['tokens_gerados']} | Tempo: {res['tempo_geracao']:.2f}s | Velocidade: {res['tps']:.2f} t/s")
                resultados.append({
                    "Modelo": modelo,
                    "Idioma": idioma,
                    "Tokens Prompt": res['tokens_prompt'],
                    "Tokens Gerados": res['tokens_gerados'],
                    "Tempo (s)": round(res['tempo_geracao'], 2),
                    "Velocidade (tokens/s)": round(res['tps'], 2)
                })
    
    if resultados:
        df = pd.DataFrame(resultados)
        print("\n--- Tabela ---")
        print(df.to_string(index=False))
        
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
        ax.set_title('Experimento 2: Comparação de Modelos (RX 580 / Vulkan)')
        ax.set_xticks(x)
        ax.set_xticklabels(modelos_unicos)
        ax.legend()
        plt.tight_layout()
        plt.savefig('../results/exp2_modelos.png')
        print("\nGráfico salvo: results/exp2_modelos.png")
        plt.close()

# ============================================================
# EXPERIMENTO 3: KV Caching (usando /api/chat para manter contexto)
# ============================================================
def experimento_kv_cache(modelo="qwen2.5:1.5b"):
    print(f"\n{'='*50}")
    print(f"EXPERIMENTO 3: KV Caching ({modelo})")
    print(f"{'='*50}")
    
    warm_up(modelo)
    resultados = []
    
    # --- COM CACHE: duas perguntas na mesma sessão de chat ---
    print("\n--- COM CACHE: Duas perguntas na mesma sessão ---")
    chat_url = "http://localhost:11434/api/chat"
    
    # Primeira mensagem (contexto longo)
    payload1 = {
        "model": modelo,
        "messages": [
            {"role": "user", "content": CONTEXTO_LONGO + "\n\n" + PERGUNTA_RELACIONADA}
        ],
        "stream": False,
        "options": {"num_predict": 50}
    }
    r1 = requests.post(chat_url, json=payload1, timeout=120)
    data1 = r1.json()
    t1 = data1.get("eval_duration", 1) / 1e9
    tps1 = data1.get("eval_count", 0) / t1 if t1 > 0 else 0
    print(f"Pergunta 1 (contexto longo): {data1.get('eval_count', 0)} tokens | {t1:.2f}s | {tps1:.2f} t/s")
    
    # Segunda mensagem (pergunta curta, mesmo contexto)
    payload2 = {
        "model": modelo,
        "messages": [
            {"role": "user", "content": CONTEXTO_LONGO + "\n\n" + PERGUNTA_RELACIONADA},
            {"role": "assistant", "content": data1.get("message", {}).get("content", "")},
            {"role": "user", "content": "Agora resuma em apenas 10 palavras."}
        ],
        "stream": False,
        "options": {"num_predict": 20}
    }
    r2 = requests.post(chat_url, json=payload2, timeout=120)
    data2 = r2.json()
    t2 = data2.get("eval_duration", 1) / 1e9
    tps2 = data2.get("eval_count", 0) / t2 if t2 > 0 else 0
    print(f"Pergunta 2 (mesmo contexto): {data2.get('eval_count', 0)} tokens | {t2:.2f}s | {tps2:.2f} t/s")
    
    resultados.append({
        "Cenário": "Pergunta 1 (contexto longo)",
        "Tokens Prompt": data1.get("prompt_eval_count", 0),
        "Tokens Gerados": data1.get("eval_count", 0),
        "Tempo (s)": round(t1, 2),
        "Velocidade (tokens/s)": round(tps1, 2)
    })
    resultados.append({
        "Cenário": "Pergunta 2 (mesmo contexto, cache)",
        "Tokens Prompt": data2.get("prompt_eval_count", 0),
        "Tokens Gerados": data2.get("eval_count", 0),
        "Tempo (s)": round(t2, 2),
        "Velocidade (tokens/s)": round(tps2, 2)
    })
    
    # --- SEM CACHE: pergunta isolada ---
    print("\n--- SEM CACHE: Pergunta isolada (sem contexto) ---")
    res = chamar_ollama(modelo, "Resuma em apenas 10 palavras o que é computação de alta performance.")
    if res:
        print(f"Tokens Prompt: {res['tokens_prompt']} | Tokens Gerados: {res['tokens_gerados']} | Tempo: {res['tempo_geracao']:.2f}s | Velocidade: {res['tps']:.2f} t/s")
        resultados.append({
            "Cenário": "Pergunta isolada (sem cache)",
            "Tokens Prompt": res['tokens_prompt'],
            "Tokens Gerados": res['tokens_gerados'],
            "Tempo (s)": round(res['tempo_geracao'], 2),
            "Velocidade (tokens/s)": round(res['tps'], 2)
        })
    
    if resultados:
        df = pd.DataFrame(resultados)
        print("\n--- Tabela ---")
        print(df.to_string(index=False))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        ax.bar(df['Cenário'], df['Velocidade (tokens/s)'], color=colors[:len(df)], alpha=0.8, width=0.5)
        ax.set_ylabel('Velocidade (tokens/s)')
        ax.set_title(f'Experimento 3: KV Caching ({modelo})')
        ax.set_ylim(0, max(df['Velocidade (tokens/s)']) * 1.3)
        
        for i, v in enumerate(df['Velocidade (tokens/s)']):
            ax.text(i, v + 1, f'{v:.2f} t/s', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../results/exp3_kv_cache.png')
        print("\nGráfico salvo: results/exp3_kv_cache.png")
        plt.close()

# ============================================================
# EXPERIMENTO 4: Quantização (Q4 vs Q8) - CORRIGIDO
# ============================================================
def experimento_quantizacao():
    print(f"\n{'='*50}")
    print("EXPERIMENTO 4: Quantização (Q4 vs Q8)")
    print(f"{'='*50}")
    
    # Tags corrigidas (Q4 usa a tag padrão)
    variantes = [
        ("qwen2.5:1.5b", "Q4_K_M (4-bit)"),
        ("qwen2.5:1.5b-instruct-q8_0", "Q8_0 (8-bit)")
    ]
    
    resultados = []
    
    for modelo_tag, label in variantes:
        print(f"\n--- Testando: {label} ({modelo_tag}) ---")
        warm_up(modelo_tag)
        
        res = chamar_ollama(modelo_tag, PROMPT_PT)
        if res:
            print(f"Tokens: {res['tokens_gerados']} | Tempo: {res['tempo_geracao']:.2f}s | Velocidade: {res['tps']:.2f} t/s")
            resultados.append({
                "Quantização": label,
                "Tokens Gerados": res['tokens_gerados'],
                "Tempo (s)": round(res['tempo_geracao'], 2),
                "Velocidade (tokens/s)": round(res['tps'], 2)
            })
        else:
            print(f"  [ERRO] Falha ao testar {modelo_tag}. Verifique se a tag existe com 'ollama list'.")
    
    if resultados:
        df = pd.DataFrame(resultados)
        print("\n--- Tabela ---")
        print(df.to_string(index=False))
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#3498db', '#e67e22']
        ax.bar(df['Quantização'], df['Velocidade (tokens/s)'], color=colors[:len(df)], alpha=0.8, width=0.5)
        ax.set_ylabel('Velocidade (tokens/s)')
        ax.set_title('Experimento 4: Impacto da Quantização (Qwen 1.5B)')
        ax.set_ylim(0, max(df['Velocidade (tokens/s)']) * 1.3)
        
        for i, v in enumerate(df['Velocidade (tokens/s)']):
            ax.text(i, v + 1, f'{v:.2f} t/s', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../results/exp4_quantizacao.png')
        print("\nGráfico salvo: results/exp4_quantizacao.png")
        plt.close()

# ============================================================
# EXPERIMENTO 4: Quantização (Q4 vs Q8)
# ============================================================
def experimento_quantizacao():
    print(f"\n{'='*50}")
    print("EXPERIMENTO 4: Quantização (Q4 vs Q8)")
    print(f"{'='*50}")
    
    # Tags de quantização do mesmo modelo
    variantes = [
        ("qwen2.5:1.5b", "Q4_K_M (4-bit)"),  
        ("qwen2.5:1.5b-instruct-q8_0", "Q8_0 (8-bit)")
    ]
    
    resultados = []
    
    for modelo_tag, label in variantes:
        print(f"\n--- Testando: {label} ---")
        warm_up(modelo_tag)
        
        res = chamar_ollama(modelo_tag, PROMPT_PT)
        if res:
            print(f"Tokens: {res['tokens_gerados']} | Tempo: {res['tempo_geracao']:.2f}s | Velocidade: {res['tps']:.2f} t/s")
            resultados.append({
                "Quantização": label,
                "Tokens Gerados": res['tokens_gerados'],
                "Tempo (s)": round(res['tempo_geracao'], 2),
                "Velocidade (tokens/s)": round(res['tps'], 2)
            })
    
    if resultados:
        df = pd.DataFrame(resultados)
        print("\n--- Tabela ---")
        print(df.to_string(index=False))
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#3498db', '#e67e22']
        ax.bar(df['Quantização'], df['Velocidade (tokens/s)'], color=colors, alpha=0.8, width=0.5)
        ax.set_ylabel('Velocidade (tokens/s)')
        ax.set_title('Experimento 4: Impacto da Quantização (Qwen 1.5B)')
        ax.set_ylim(0, max(df['Velocidade (tokens/s)']) * 1.3)
        
        for i, v in enumerate(df['Velocidade (tokens/s)']):
            ax.text(i, v + 1, f'{v:.2f} t/s', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../results/exp4_quantizacao.png')
        print("\nGráfico salvo: results/exp4_quantizacao.png")
        plt.close()

# ============================================================
# MAIN: Roda todos os experimentos
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK COMPLETO: LLMs Locais com Ollama + Vulkan")
    print("Hardware: AMD RX 580 2048SP (8GB VRAM)")
    print("=" * 60)
    
    experimento_idiomas("qwen2.5:1.5b")
    experimento_modelos()
    experimento_kv_cache("qwen2.5:1.5b")
    experimento_quantizacao()
    
    print("\n" + "=" * 60)
    print("TODOS OS EXPERIMENTOS CONCLUÍDOS!")
    print("Gráficos salvos em: results/")
    print("=" * 60)