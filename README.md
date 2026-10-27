## Resultados

### Experimento 1: PT-BR vs Inglês (Ollama + Vulkan)

**Prompt PT-BR:** "Explique em uma frase o que é computação de alta performance."
**Prompt EN:** "Explain in one sentence what high-performance computing is."

| Métrica | PT-BR | EN | Diferença |
| :--- | :--- | :--- | :--- |
| Tokens do Prompt | 43 | 40 | -7% no inglês |
| Tokens Gerados | 50 | 36 | -28% no inglês |
| Tempo Total | 1.26s | 0.91s | **-28% no inglês** |
| Velocidade | 39.60 t/s | 39.55 t/s | Praticamente igual |

**Insights:**
- O prompt em português consumiu mais tokens de entrada (43 vs 40).
- O modelo gerou **28% menos tokens** em inglês para concluir a mesma tarefa.
- O tempo total em inglês foi **28% menor**. Em cenários de alto volume, isso representa uma economia significativa.

### Experimento 2: Comparação de Modelos

Todos os testes com warm-up (modelo já carregado na VRAM) para medir apenas o tempo de geração.

| Modelo | Parâmetros | Velocidade PT-BR (t/s) | Velocidade EN (t/s) | Tokens Gerados PT-BR | Tokens Gerados EN |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen 2.5 1.5B | 1.5B | **39.64** | **40.47** | 50 | 31 |
| Qwen 2.5 3B | 3B | 29.02 | 29.17 | 36 | 27 |
| Llama 3.2 1B | 1B | 22.11 | 22.45 | 50 | 38 |

**Insights:**
- O **Qwen 2.5 1.5B** apresentou o melhor equilíbrio entre velocidade e qualidade, sendo ~80% mais rápido que o Llama 3.2 1B, apesar de ter mais parâmetros.
- O **tamanho do modelo não é o único fator de velocidade**: arquitetura e otimização do backend influenciam significativamente.
- A velocidade de geração é praticamente idêntica entre PT-BR e EN para o mesmo modelo.

### Experimento 3: KV Caching

| Cenário | Tokens Prompt | Tokens Gerados | Tempo (s) | Velocidade (t/s) |
| :--- | :--- | :--- | :--- | :--- |
| Pergunta 1 (contexto longo) | 203 | 28 | 0.70 | 40.16 |
| **Pergunta 2 (mesmo contexto, cache)** | **253** | **13** | **0.32** | **40.01** |
| Pergunta isolada (sem cache) | 47 | 23 | 0.58 | 39.36 |

**Insight:** A segunda pergunta no mesmo contexto levou **0.32s** — quase **metade do tempo** da pergunta isolada (0.58s), mesmo tendo um prompt **5x maior** (253 vs 47 tokens). Isso prova que o KV Caching reaproveita os cálculos do contexto anterior.

### Experimento 4: Quantização (Q4 vs Q8)

| Quantização | Tamanho | Tokens Gerados | Tempo (s) | Velocidade (t/s) |
| :--- | :--- | :--- | :--- | :--- |
| Q4_K_M (4-bit) | ~986 MB | 50 | 1.26 | **39.79** |
| Q8_0 (8-bit) | ~1.6 GB | 50 | 1.37 | 36.37 |

**Insight:** O Q4 é **~9% mais rápido** que o Q8 e ocupa **quase metade do espaço em disco/VRAM**. Para hardware limitado como a RX 580, o Q4 é a escolha ideal.

### Experimento 5: CPU vs GPU (Vulkan)

| Configuração | Velocidade (tokens/s) |
| :--- | :--- |
| CPU (PyTorch) | ~7 |
| GPU (Vulkan) | **~40** |

**Ganho:** Aproximadamente **6x** de speedup apenas ao ativar a GPU via Vulkan.

### Experimento 6: Teste de Qualidade (Jarvis 1.5B vs 3B)

Criamos dois modelos customizados (`jarvis` baseado no Qwen 1.5B e `jarvis-3b` baseado no Qwen 3B) com o mesmo system prompt de engenharia. Testamos a mesma pergunta técnica em ambos:

**Pergunta:** "Explique a diferença entre um microcontrolador e um microprocessador."

**Resposta do Jarvis 1.5B:**
> Descreveu o microprocessador como tendo "memória e hardware de entrada/saída" integrados — **incorreto**. O microprocessador é apenas a CPU.

**Resposta do Jarvis 3B:**
> Acertou que o microprocessador é a CPU, mas afirmou que "o microprocessador contém um microcontrolador" — **inversão conceitual**. O correto é o contrário.

**Insight:** Mesmo modelos maiores (3B) podem cometer erros conceituais em domínios técnicos específicos. O system prompt define o tom, mas **não corrige alucinações**. Isso justifica a necessidade de **fine-tuning com QLoRA** usando dados corretos e específicos do domínio.

### Experimento 7: Fine-Tuning com QLoRA (Jarvis LoRA)

**Objetivo:** Criar um assistente especializado em mecatrônica usando fine-tuning eficiente (QLoRA) em hardware limitado.

**Pipeline:**
1. Treinamento do adaptador LoRA no Google Colab (GPU T4 gratuita) com dataset de 5 exemplos de mecatrônica.
2. Merge do adaptador no modelo base Qwen 2.5 1.5B (f16).
3. Conversão para GGUF via `llama.cpp`.
4. Importação no Ollama como `jarvis-lora` (3.1 GB).

**Desafios encontrados e soluções:**
| Desafio | Solução |
| :--- | :--- |
| `bitsandbytes` não funciona na RX 580 | Treinamento realizado no Colab (GPU T4) |
| Erro de `bf16` no GradScaler da T4 | Conversão dos parâmetros treináveis para `float32` após criar o `SFTTrainer` |
| Conflito de versão do `torchao` no Colab | Remoção da biblioteca (`pip uninstall torchao`) |
| Tokenizador do Qwen incompatível com `convert_hf_to_gguf.py` | Download do `tokenizer.json` e `tokenizer_config.json` originais do Hugging Face |
| `ADAPTER` removido do Ollama (v0.34.1+) | Merge do adaptador no modelo base antes da conversão para GGUF |

**Resultado:** Pipeline completo de fine-tuning validado, desde o treinamento até a inferência local. O modelo fine-tunado apresentou mudança no padrão de resposta em comparação ao modelo base, confirmando que o adaptador teve efeito.

**Limitações:** Com apenas 5 exemplos e 5 épocas, a melhoria é sutil. Para resultados robustos, seria necessário um dataset maior (50-100 exemplos) e mais épocas de treinamento.

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