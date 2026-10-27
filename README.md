# Benchmark de LLMs Locais: Otimização e Eficiência em Hardware Limitado

## Visão Geral

Este projeto investiga o impacto de **técnicas de otimização de inferência** no desempenho de Large Language Models (LLMs) executados localmente em hardware de consumidor. O foco está em dois eixos principais:

1. **Eficiência de Hardware:** Como diferentes backends de execução (CPU vs. GPU via Vulkan) afetam a velocidade de geração de tokens.
2. **Eficiência de Tokenização:** Como o idioma do prompt (Português vs. Inglês) influencia o custo computacional e o tempo de resposta.

O objetivo é demonstrar, com dados reais, que é possível rodar LLMs de forma eficiente em placas de vídeo antigas (como a AMD RX 580), contornando limitações de drivers e ecossistemas fechados.

## Contexto e Motivação

A maioria dos tutoriais de LLMs assume acesso a GPUs NVIDIA modernas com CUDA. No entanto, grande parte dos desenvolvedores e entusiastas possui hardware mais antigo ou de outras marcas.

Este projeto nasceu da necessidade prática de responder: **"É possível rodar um LLM de forma útil em uma RX 580 de 8GB?"**

A resposta é sim, mas o caminho não é o padrão. Este repositório documenta a jornada: desde a falha do PyTorch/CUDA, passando pela instabilidade do ROCm, até a solução estável com **Ollama + Vulkan**.

## Decisões Técnicas

### Por que Ollama + Vulkan?

- **PyTorch/CUDA:** Inviável. CUDA é exclusivo para GPUs NVIDIA.
- **ROCm:** O suporte oficial da AMD para placas Polaris (RX 580) foi descontinuado. Tentativas de usar `bitsandbytes` com ROCm no Windows resultaram em loops infinitos e instabilidade.
- **Vulkan:** A API gráfica multiplataforma é suportada nativamente pela RX 580. O Ollama, utilizando o backend `llama.cpp`, consegue acessar a GPU via Vulkan de forma estável.

**Resultado:** Salto de ~7 tokens/s (CPU) para **~40 tokens/s (GPU via Vulkan)**.

### Por que comparar PT-BR vs Inglês?

Modelos como o Qwen são treinados majoritariamente em inglês e chinês. A tokenização de textos em português tende a gerar mais tokens para expressar a mesma ideia, aumentando o custo (tempo e, em APIs pagas, dinheiro). O benchmark quantifica essa diferença.

## Hardware Utilizado

| Componente | Especificação |
| :--- | :--- |
| GPU | AMD Radeon RX 580 2048SP (8GB VRAM) |
| Backend | Ollama + Vulkan (llama.cpp) |
| Modelo | Qwen 2.5 1.5B (Q4_K_M) |
| Sistema | Windows 10/11 |

## Resultados

### Experimento 1: PT-BR vs Inglês (Ollama + Vulkan)

**Prompt PT-BR:** "Explique em uma frase o que é computação de alta performance."
**Prompt EN:** "Explain in one sentence what high-performance computing is."

| Métrica | PT-BR | EN | Diferença |
| :--- | :--- | :--- | :--- |
| Tokens do Prompt | 43 | 40 | **-7% no inglês** |
| Tokens Gerados | 50 | 27 | **-46% no inglês** |
| Tempo Total | 1.41s | 0.84s | **-40% no inglês** |
| Velocidade | 40.35 t/s | 40.89 t/s | Praticamente igual |

**Insights:**
- O prompt em português consumiu **mais tokens de entrada** (43 vs 40), confirmando a ineficiência do tokenizador para o português.
- O modelo gerou **quase o dobro de tokens** em português, indicando maior "verbosidade" ou dificuldade em encerrar o raciocínio.
- O tempo total em inglês foi **40% menor**. Em cenários de alto volume, isso representa uma economia significativa.

### Experimento 2: CPU vs GPU (Vulkan)

| Configuração | Velocidade (tokens/s) |
| :--- | :--- |
| CPU (PyTorch) | ~7 |
| GPU (Vulkan) | **~40** |

**Ganho:** Aproximadamente **6x** de speedup apenas ao ativar a GPU via Vulkan.

## Como Executar

### Pré-requisitos
- Python 3.10+
- Ollama instalado
- GPU AMD com suporte a Vulkan (ou NVIDIA/Intel)

### Passos

1. Clone o repositório:
   ```bash
   git clone https://github.com/JPauloNGA/llm_study.git
   cd llm_study
   ```

2. Crie e ative o ambiente virtual:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Baixe o modelo no Ollama:
   ```bash
   ollama pull qwen2.5:1.5b
   ```

5. (Opcional) Ative o Vulkan para GPUs AMD antigas:
   ```powershell
   [System.Environment]::SetEnvironmentVariable('OLLAMA_VULKAN', '1', 'User')
   # Reinicie o Ollama
   ```

6. Execute o benchmark:
   ```bash
   python src/benchmark.py
   ```

## Próximos Passos (Roadmap)

- [ ] Testar outros modelos (Llama 3.2 1B, Gemma 2 2B)
- [ ] Comparar diferentes níveis de quantização (Q4, Q5, Q8)
- [ ] Medir uso real de VRAM com ferramentas do sistema
- [ ] Adicionar teste de qualidade (perplexidade) para validar se a quantização degrada a resposta

## Tecnologias

- [Ollama](https://ollama.com/) - Runtime de LLM local
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Engine de inferência (backend do Ollama)
- [Vulkan](https://www.vulkan.org/) - API gráfica
- Python, Pandas, Matplotlib

## Licença

MIT