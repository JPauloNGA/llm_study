import requests
import time
import pandas as pd
import matplotlib.pyplot as plt

OLLAMA_URL = "http://localhost:11434/api/generate"

TAREFA = """Escreva uma função Python chamada `soma_quadrados_pares` que receba uma lista de inteiros e retorne a soma dos quadrados dos números pares, usando list comprehension. Inclua um exemplo de uso."""

MODELOS = {
    "Especialista (Coder 7B)": "qwen2.5-coder:7b",
    "Generalista (Qwen 7B)": "qwen2.5:7b"
}

def warm_up(modelo):
    print(f"  [Warm-up] Carregando {modelo}...")
    payload = {"model": modelo, "prompt": "OK", "stream": False, "options": {"num_predict": 1}}
    try:
        requests.post(OLLAMA_URL, json=payload, timeout=120)
    except Exception as e:
        print(f"  Erro no warm-up: {e}")

def rodar_teste(modelo_tag, label):
    print(f"\n--- Testando: {label} ({modelo_tag}) ---")
    warm_up(modelo_tag)
    
    payload = {
        "model": modelo_tag,
        "prompt": TAREFA,
        "stream": False,
        "options": {"num_predict": 300}
    }
    
    start = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    except requests.exceptions.Timeout:
        print("Timeout.")
        return None
    end = time.time()
    
    if response.status_code != 200:
        print(f"Erro: {response.status_code}")
        return None
    
    data = response.json()
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 1) / 1e9
    tps = eval_count / eval_duration if eval_duration > 0 else 0
    tempo_total = end - start
    
    print(f"Tempo total: {tempo_total:.2f}s | Tokens: {eval_count} | Velocidade: {tps:.2f} t/s")
    print(f"Resposta:\n{data.get('response', '')[:500]}...")
    
    return {
        "Modelo": label,
        "Tag": modelo_tag,
        "Tempo Total (s)": round(tempo_total, 2),
        "Tokens Gerados": eval_count,
        "Velocidade (t/s)": round(tps, 2),
        "Resposta": data.get('response', '')
    }

if __name__ == "__main__":
    resultados = []
    for label, tag in MODELOS.items():
        res = rodar_teste(tag, label)
        if res:
            resultados.append(res)
    
    if resultados:
        df = pd.DataFrame(resultados)
        print("\n" + "="*60)
        print("RESUMO")
        print("="*60)
        print(df[["Modelo", "Tempo Total (s)", "Tokens Gerados", "Velocidade (t/s)"]].to_string(index=False))
        
        # Gráfico comparativo
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(df['Modelo'], df['Velocidade (t/s)'], color=colors, alpha=0.8, width=0.5)
        ax.set_ylabel('Velocidade (tokens/s)')
        ax.set_title('Experimento 9: Especialista vs. Generalista (RX 580)')
        ax.set_ylim(0, max(df['Velocidade (t/s)']) * 1.3)
        
        for i, v in enumerate(df['Velocidade (t/s)']):
            ax.text(i, v + 0.5, f'{v:.2f} t/s', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../results/exp9_especialistas.png')
        print("\nGráfico salvo: results/exp9_especialistas.png")
        plt.show()
        
        # Salvar respostas completas para análise manual
        with open('../results/exp9_respostas.txt', 'w', encoding='utf-8') as f:
            for r in resultados:
                f.write(f"=== {r['Modelo']} ===\n")
                f.write(r['Resposta'])
                f.write("\n\n")
        print("Respostas completas salvas em: results/exp9_respostas.txt")