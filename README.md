# Análise Quantitativa do Trade-off entre Especialização e Generalização em LLMs

## 1. Visão Geral do Projeto

Este repositório contém o código e os resultados de um projeto experimental para avaliar o processo de fine-tuning em Modelos de Linguagem de Grande Porte (LLMs). O objetivo foi implementar, treinar e avaliar um LLM para a tarefa de Text-to-SQL, quantificando o ganho de desempenho na tarefa-alvo e, simultaneamente, medindo a degradação de performance (esquecimento catastrófico) em tarefas de conhecimento geral.

- **Modelo Base:** `google/gemma-2b-it` (substituído dos modelos de 7B devido a limitações técnicas do ambiente Colab)
- **Tarefa de Especialização:** Text-to-SQL usando o dataset Spider.
- **Avaliação de Generalização:** Benchmark MMLU em 3 categorias (STEM, Humanidades, Ciências Sociais).
- **Técnica de Fine-Tuning:** PEFT (LoRA).

## 2. Estrutura do Repositório

```
.
├── custom_metrics/
│   └── execution_accuracy.py   # Código da métrica customizada para DeepEval
├── scripts/
│   ├── train.py                # Script para treinar os modelos
│   └── evaluate.py             # Script para avaliar os modelos e gerar resultados
├── .gitignore
├── README.md                   # Este arquivo
└── requirements.txt            # Dependências do projeto
```

## 3. Configuração do Ambiente

1.  **Clone o Repositório:**
    ```bash
    git clone [URL_DO_SEU_REPOSITORIO]
    cd [NOME_DO_SEU_REPOSITORIO]
    ```

2.  **Crie um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Autenticação no Hugging Face:**
    Você precisará de um token de acesso do Hugging Face para baixar o modelo.
    ```bash
    huggingface-cli login
    # Cole seu token quando solicitado
    ```

5.  **Dados do Projeto:**
    Baixe o **Spider dataset** (`spider.zip` do site oficial ou a fonte que você usou) e renomeie o arquivo para `spider_data.zip`. Coloque-o na **raiz** do seu repositório clonado.

## 4. Como Reproduzir os Resultados

Os scripts são projetados para serem executados a partir da raiz do repositório.

### Passo 1: Treinamento

Execute o script de treinamento para cada um dos dois experimentos. O script irá descompactar os dados e salvar os adaptadores LoRA treinados em pastas de saída (ex: `gemma-2b-spider-ft-exp1/`).

```bash
# Treinar o Experimento 1 (learning rate = 2e-4)
python scripts/train.py --experiment_num 1

# Treinar o Experimento 2 (learning rate = 5e-5)
python scripts/train.py --experiment_num 2
```

### Passo 2: Avaliação

Após os dois modelos terem sido treinados, execute o script de avaliação. Ele carregará o modelo base e os dois adaptadores LoRA salvos, e rodará todas as avaliações (Execution Accuracy e MMLU), imprimindo a tabela de resultados final no console.

```bash
python scripts/evaluate.py
```

A saída será uma tabela formatada com todas as métricas necessárias para a análise do projeto.
