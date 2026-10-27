import requests
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

ESPECIALISTAS = {
    "codigo": "qwen2.5-coder:7b",
    "mecatronica": "jarvis",
    "visao": "llava:7b",
    "geral": "qwen2.5:3b"
}

PROMPT_CLASSIFICADOR = """Você é um classificador de perguntas. Leia a pergunta e classifique em UMA das categorias abaixo.
Responda APENAS com a palavra da categoria, sem explicações, sem pontuação.

Categorias e regras:
- codigo: perguntas sobre programação, algoritmos, debugging, linguagens de programação, funções, bibliotecas.
- mecatronica: perguntas sobre eletrônica, robótica, automação, sistemas embarcados, motores, sensores, atuadores, microcontroladores.
- visao: perguntas que envolvem análise de imagens, reconhecimento visual, ou que precisam "ver" algo.
- geral: qualquer outra pergunta (história, ciência, filosofia, conhecimentos gerais).

Exemplos:
Pergunta: "Como inverter uma lista em Python?" -> codigo
Pergunta: "O que é um sensor de efeito Hall?" -> mecatronica
Pergunta: "Quem foi Alan Turing?" -> geral
Pergunta: "Descreva a imagem que estou enviando." -> visao
Pergunta: "Qual a diferença entre servo motor e motor de passo?" -> mecatronica

Pergunta do usuário: {pergunta}

Categoria:"""

def warm_up(modelo):
    """Carrega o modelo na VRAM antes de medir."""
    payload = {"model": modelo, "prompt": "OK", "stream": False, "options": {"num_predict": 1}}
    try:
        requests.post(OLLAMA_URL, json=payload, timeout=120)
    except Exception:
        pass

def classificar(pergunta, modelo_classificador="qwen2.5:1.5b"):
    prompt = PROMPT_CLASSIFICADOR.format(pergunta=pergunta)
    payload = {
        "model": modelo_classificador,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 5, "temperature": 0.0}
    }
    
    start = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        return "geral", time.time() - start, "timeout"
    end = time.time()
    
    if response.status_code != 200:
        return "geral", end - start, "erro"
    
    data = response.json()
    categoria = data.get("response", "").strip().lower()
    categoria = categoria.replace(".", "").replace(",", "").replace("\n", "").strip()
    
    if categoria not in ESPECIALISTAS:
        categoria = "geral"
    
    return categoria, end - start, data.get("response", "")

def rotear(pergunta):
    print(f"\n--- Pergunta: {pergunta} ---")
    
    categoria, tempo_classificacao, resposta_bruta = classificar(pergunta)
    print(f"[Classificador] Categoria: {categoria} (tempo: {tempo_classificacao:.2f}s)")
    print(f"[Classificador] Resposta bruta: '{resposta_bruta}'")
    
    modelo_especialista = ESPECIALISTAS[categoria]
    print(f"[Roteador] Chamando especialista: {modelo_especialista}")
    
    payload = {
        "model": modelo_especialista,
        "prompt": pergunta,
        "stream": False,
        "options": {"num_predict": 100}
    }
    
    start = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    except requests.exceptions.Timeout:
        print(f"[Erro] Timeout ao chamar {modelo_especialista}")
        return None
    end = time.time()
    
    if response.status_code != 200:
        print(f"[Erro] Falha ao chamar {modelo_especialista}")
        return None
    
    data = response.json()
    tempo_especialista = end - start
    tokens_gerados = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 1) / 1e9
    tps = tokens_gerados / eval_duration if eval_duration > 0 else 0
    
    print(f"[Especialista] Tempo total: {tempo_especialista:.2f}s")
    print(f"[Especialista] Tokens gerados: {tokens_gerados} | Velocidade: {tps:.2f} t/s")
    print(f"[Resposta] {data.get('response', '')[:300]}...")
    
    return {
        "pergunta": pergunta,
        "categoria": categoria,
        "modelo_especialista": modelo_especialista,
        "tempo_classificacao": round(tempo_classificacao, 2),
        "tempo_especialista": round(tempo_especialista, 2),
        "tempo_total": round(tempo_classificacao + tempo_especialista, 2),
        "tokens_gerados": tokens_gerados,
        "velocidade": round(tps, 2)
    }

if __name__ == "__main__":
    print("Fazendo warm-up do classificador...")
    warm_up("qwen2.5:1.5b")
    print("Warm-up concluído.")
    
    perguntas_teste = [
        "Como faço para inverter uma lista em Python?",
        "Qual a diferença entre um servo motor e um motor de passo?",
        "Explique o que é um sensor de efeito Hall.",
        "Quem foi Alan Turing?",
        "Descreva a imagem que estou enviando.",
    ]
    
    resultados = []
    for pergunta in perguntas_teste:
        res = rotear(pergunta)
        if res:
            resultados.append(res)
    
    print("\n" + "="*60)
    print("RESUMO DO ROTEAMENTO")
    print("="*60)
    for r in resultados:
        print(f"[{r['categoria']:12s}] -> {r['modelo_especialista']:20s} | Classificação: {r['tempo_classificacao']}s | Total: {r['tempo_total']}s | {r['velocidade']} t/s")