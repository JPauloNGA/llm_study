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

### Experimento 8: Roteador de Especialistas (Router)

**Tese:** A inteligência pode emergir da composição de modelos especializados pequenos, em vez de um único modelo monolítico gigante.

**Arquitetura:**
1. **Classificador:** Qwen 2.5 1.5B (rápido, ~2.3s) lê a pergunta e decide a categoria.
2. **Roteador:** Com base na categoria, chama o especialista certo via API do Ollama.
3. **Especialistas:**
   - `codigo` → Qwen 2.5 Coder 7B
   - `mecatronica` → Jarvis (Qwen 1.5B com system prompt técnico)
   - `visao` → Llava 7B
   - `geral` → Qwen 2.5 3B (fallback)

**Resultados (5 perguntas de teste):**

| Pergunta | Categoria | Especialista | Tempo Classificação | Tempo Total | Velocidade |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Inverter lista em Python | codigo | qwen2.5-coder:7b | 2.94s | 12.22s | 14.73 t/s |
| Servo motor vs motor de passo | mecatronica | jarvis | 2.33s | 7.41s | 38.92 t/s |
| Sensor de efeito Hall | mecatronica | jarvis | 2.36s | 7.20s | 38.52 t/s |
| Quem foi Alan Turing? | geral | qwen2.5:3b | 2.28s | 23.64s | 28.66 t/s |
| Descreva a imagem | visao | llava:7b | 14.06s | 41.68s | 22.98 t/s |

**Insights:**
- O classificador acertou **100%** das categorias com o prompt corrigido.
- O **Jarvis (1.5B especializado)** foi o especialista mais rápido (~7s total, 38.92 t/s), superando modelos maiores como o Coder 7B e o Llama 7B.
- Modelos 7B (Coder e Llava) são lentos na RX 580 (8GB VRAM) porque o modelo precisa ser carregado/descarregado a cada chamada, adicionando overhead.
- **Conclusão parcial:** A tese de composição de especialistas é viável, mas o hardware limita o tamanho dos especialistas. Modelos pequenos (1.5B) especializados são a melhor opção para a RX 580.

**Limitações:**
- O Llava 7B recusou a pergunta de visão porque nenhuma imagem foi enviada (comportamento esperado).
- O tempo de classificação da pergunta de visão foi alto (14s) porque o Llava precisou ser carregado na VRAM pela primeira vez.

### Experimento 9: Especialista vs. Generalista (Qualidade e Velocidade)

**Tese:** Um modelo especialista pequeno supera um generalista maior em tarefas específicas.

**Tarefa:** "Escreva uma função Python que receba uma lista de inteiros e retorne a soma dos quadrados dos números pares, usando list comprehension."

| Modelo | Tempo Total | Tokens Gerados | Velocidade | Qualidade |
| :--- | :--- | :--- | :--- | :--- |
| **Especialista (Coder 7B)** | 19.70s | 255 | 15.08 t/s | Código idiomático, docstring completa, generator expression (eficiente em memória) |
| **Generalista (Qwen 7B)** | 17.49s | 300 | 20.42 t/s | Código funcional, mas usa list comprehension dentro do `sum` (cria lista intermediária), sem docstring |

**Insights:**
- O generalista foi mais rápido (20.42 vs 15.08 t/s), mas gerou **mais tokens** (300 vs 255) e um código **menos eficiente** em memória.
- O especialista gerou um código **mais idiomático e otimizado**, com docstring completa.
- **Conclusão:** A velocidade bruta não é o único critério. O especialista entrega **mais qualidade por token gerado**, o que é crucial em sistemas de produção onde a qualidade da resposta importa mais que a velocidade bruta.

**Limitações:**
- Ambos são modelos 7B, então o custo de VRAM é similar. A vantagem do especialista está na qualidade, não no custo.
- Para tarefas muito simples, o generalista pode ser suficiente. O especialista brilha em tarefas complexas que exigem conhecimento de domínio.

### Experimento 10: RAG (Retrieval-Augmented Generation)

**Tese:** Um modelo pequeno (1.5B) com acesso a uma base de conhecimento externa (RAG) supera o mesmo modelo sem RAG em precisão, sem custo significativo de tempo.

**Arquitetura:**
1. Documentos de mecatrônica indexados com `nomic-embed-text` (embeddings).
2. Busca por similaridade de cosseno para encontrar os trechos mais relevantes.
3. Contexto injetado no prompt do Jarvis antes da geração.

**Resultados:**

| Pergunta | Sem RAG (tempo) | Com RAG (busca) | Com RAG (geração) | Com RAG (total) | Sem RAG (t/s) | Com RAG (t/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Microcontrolador vs microprocessador | 6.15s | 2.12s | 4.36s | 6.48s | 38.62 | 38.69 |
| O que é servo motor? | 6.13s | 2.14s | 4.78s | 6.92s | 38.56 | 38.44 |

**Insights:**
- **Tempo total:** O RAG adicionou apenas ~0.3s a 0.8s ao tempo total. A busca (2.1s) foi compensada pelo fato de o Jarvis gerar menos tokens com RAG (85 e 101 vs 150 sem RAG).
- **Qualidade:** A resposta com RAG foi **direta e tecnicamente correta** ("o microprocessador é apenas a CPU..."). Sem RAG, a resposta foi mais genérica e propensa a erros conceituais.
- **Eficiência:** O Jarvis com RAG gerou **menos tokens** (mais direto) e manteve a mesma velocidade de geração (~38.5 t/s).
- **Conclusão:** O RAG permite que um modelo pequeno (1.5B) tenha **conhecimento de domínio preciso** sem precisar ser treinado nisso. Isso valida a tese de que **especialistas pequenos + base de conhecimento compartilhada** são uma alternativa viável a modelos monolíticos gigantes.

**Limitações:**
- A base de conhecimento atual tem apenas 1 documento (microcontroladores). Para resultados robustos, seria necessário indexar mais documentos (motores, sensores, etc.).
- A busca por similaridade de cosseno é simples; para bases maiores, seria necessário um banco vetorial dedicado (ex: ChromaDB, FAISS).

### Experimento 11: Comparação Estendida de Modelos (Qwen 1.5B, Llama 1B, Gemma 2B, Qwen 3B)

**Objetivo:** Comparar velocidade, uso de VRAM e eficiência de tokenização entre modelos de diferentes tamanhos e arquiteturas.

| Modelo | Parâmetros | Velocidade (t/s) | VRAM (GB) | Tokens Prompt PT-BR | Tokens Prompt EN |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 1.5B** | 1.5B | **~39** | ~1.0 | 43 | 40 |
| **Llama 3.2 1B** | 1B | ~22.5 | 1.41 | 39 | 36 |
| **Gemma 2 2B** | 2B | ~28.4 | 1.80 | **23** | **20** |
| **Qwen 2.5 3B** | 3B | ~29.2 | 2.01 | 43 | 40 |

**Insights:**
- **Arquitetura > Tamanho:** O Qwen 1.5B (~39 t/s) é quase 2x mais rápido que o Llama 1B (~22.5 t/s), apesar de ter mais parâmetros. Isso mostra que a arquitetura e a otimização do backend (llama.cpp) importam mais que o número de parâmetros.
- **Gemma 2B é o "meio-termo" ideal:** ~28.4 t/s com apenas 1.8 GB de VRAM, quase empatando com o Qwen 3B (~29.2 t/s) mas usando menos memória.
- **Tokenizador do Gemma é muito mais eficiente em PT-BR:** Apenas **23 tokens** para o prompt em português, contra 43 do Qwen. Isso significa que o Gemma "entende" português com menos tokens, reduzindo o custo computacional em cenários multilíngues.
- **A diferença PT-BR vs EN se mantém:** Em todos os modelos, o prompt em inglês gerou menos tokens e foi mais rápido. O Gemma foi a exceção (tempo similar), provavelmente porque seu tokenizador é mais equilibrado entre idiomas.
- **Insight extra:** O Qwen 1.5B é o modelo mais eficiente para a RX 580 (melhor velocidade com menor VRAM). O Llama 3.2 1B, apesar de ter menos parâmetros, é mais lento e usa mais VRAM que o Qwen 1.5B — provando que arquitetura e tokenizador importam mais que o número de parâmetros.

### Experimento 12: Impacto da Quantização (Q4 vs Q8)

**Objetivo:** Comparar o trade-off entre velocidade, VRAM e qualidade das quantizações Q4_K_M (4-bit) e Q8_0 (8-bit) do mesmo modelo (Qwen 2.5 1.5B).

| Quantização | Velocidade (t/s) | VRAM (GB) | Tempo (s) | Observação |
| :--- | :--- | :--- | :--- | :--- |
| **Q4_K_M (4-bit)** | **38.79** | **1.09** | 1.29 | ~5.6% mais rápido, 36% menos VRAM |
| **Q8_0 (8-bit)** | 36.74 | 1.70 | 1.36 | Resposta com alucinação ("computação em tempo real") |

**Insights:**
- O Q4 é **~5.6% mais rápido** e usa **36% menos VRAM** que o Q8. Para a RX 580 (8GB), isso é crucial: libera VRAM para carregar outros modelos simultaneamente (ex: roteador + especialista).
- **Mais bits não significam melhor qualidade:** o Q8 alucinou ao definir HPC como "computação em tempo real", enquanto o Q4 acertou ("HPC - High Performance Computing"). Isso mostra que a quantização não é o único fator de qualidade; o treinamento e o alinhamento do modelo importam mais.
- **Conclusão:** Para hardware limitado como a RX 580, o Q4 é a escolha ideal. O Q8 só compensa se a precisão for absolutamente crítica e houver VRAM sobrando (o que não é o caso).

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

- [✓] Testar outros modelos (Llama 3.2 1B, Gemma 2 2B)
- [✓] Comparar diferentes níveis de quantização (Q4, Q8)
- [ ] Medir uso real de VRAM com ferramentas do sistema
- [ ] Adicionar teste de qualidade (perplexidade) para validar se a quantização degrada a resposta

## Tecnologias

- [Ollama](https://ollama.com/) - Runtime de LLM local
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Engine de inferência (backend do Ollama)
- [Vulkan](https://www.vulkan.org/) - API gráfica
- Python, Pandas, Matplotlib

## Licença

MIT