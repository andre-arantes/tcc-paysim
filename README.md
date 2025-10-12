# Detecção de Fraudes em Transações com Spark Streaming

Este projeto implementa um pipeline completo para detecção de fraudes em tempo real utilizando o dataset PaySim. A arquitetura é baseada em Apache Kafka para ingestão de dados e Apache Spark (Structured Streaming) para processamento e inferência com modelos de Machine Learning como Regressão Logística, Random Forest e XGBoost.

## Requisitos

- Python 3.10+
- Java 8 ou 11 (`java -version`)
- Apache Spark 4.0+
- Apache Kafka 2.8+
- [Homebrew](https://brew.sh/) (para usuários de macOS, para instalar dependências como `libomp`)

## ⚙️ Configuração do Ambiente (Setup)

Siga estes passos para configurar o ambiente de desenvolvimento local.

1.  **Clonar o Repositório**
    ```bash
    git clone <url-do-seu-repositorio>
    cd tcc-paysim
    ```

2.  **Criar e Ativar o Ambiente Virtual**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # No Windows, use: venv\Scripts\activate
    ```

3.  **Instalar as Dependências Python**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Baixar o Dataset**
    - Faça o download do dataset PaySim no Kaggle: [Synthetic Financial Dataset For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
    - Descompacte o arquivo e coloque o `PS_20174392719_1491204439457_log.csv` dentro da pasta `data/raw/`.

## 🚀 Como Executar o Projeto

A execução é dividida em duas fases principais: o treinamento offline dos modelos e a simulação da inferência em tempo real.

### Fase 0: Configuração do Kafka

Antes de iniciar o streaming, o Kafka precisa estar rodando. Estes comandos devem ser executados a partir do diretório de instalação do Kafka.

1.  **Iniciar o Zookeeper** (em um terminal)
    ```bash
    bin/zookeeper-server-start.sh config/zookeeper.properties
    ```

2.  **Iniciar o Servidor Kafka** (em um segundo terminal)
    ```bash
    /opt/homebrew/opt/kafka/bin/kafka-server-start /opt/homebrew/etc/kafka/server.properties
    ```

3.  **Criar o Tópico `transactions`** (em um terceiro terminal)
    - Este comando só precisa ser executado uma vez.
    ```bash
    bin/kafka-topics.sh --create --topic transactions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
    ```

### Fase 1: Pipeline Offline (Treinamento dos Modelos)

Estes scripts preparam os dados e treinam os modelos de Machine Learning. Devem ser executados a partir da **pasta raiz do projeto (`tcc-paysim/`)**.

1.  **Criar o Subconjunto de Dados (2M de registros)**
    ```bash
    python src/pipelines/1_create_subset.py
    ```

2.  **Executar o Pré-processamento Pesado**
    - Este script prepara os dados para todos os modelos e salva o resultado. Ele só precisa ser executado uma vez.
    - *Nota: Você pode editar a flag `USE_SUSPICIOUS_FEATURE` dentro do script para gerar dados com ou sem a feature de engenharia.*
    ```bash
    python src/pipelines/2_preprocess_data.py
    ```

3.  **Treinar os Modelos**
    - Execute cada script de treinamento separadamente. Isso evita sobrecarregar sua máquina.
    ```bash
    # Treinar o Random Forest
    python src/pipelines/3_train_rf.py

    # Treinar a Regressão Logística
    python src/pipelines/4_train_lr.py

    # Treinar o XGBoost
    python src/pipelines/5_train_xgb.py
    ```

4.  **(Opcional) Interpretar os Modelos Treinados**
    - Este script carrega os modelos salvos e exibe a importância das features ou os coeficientes.
    ```bash
    python src/pipelines/6_interpret_models.py
    ```

### Fase 2: Pipeline em Tempo Real (Inferência com Streaming)

Esta fase simula a detecção de fraudes em tempo real. Você precisará de pelo menos dois terminais abertos, ambos na **pasta raiz do projeto**.

1.  **Iniciar o Produtor de Dados (Terminal 1)**
    - Este script começará a ler o arquivo CSV e a enviar transações para o tópico `transactions` do Kafka.
    ```bash
    python src/streaming/kafka_producer.py
    ```

2.  **Iniciar o Consumidor com o Modelo Desejado (Terminal 2)**
    - O consumidor irá receber as transações, aplicar o modelo e exibir um alerta no console sempre que uma fraude for detectada.
    ```bash
    # Exemplo para o modelo Random Forest
    python src/streaming/consumer_rf.py

    # Para rodar com outro modelo, basta chamar o script correspondente:
    # python src/streaming/consumer_lr.py
    # python src/streaming/consumer_xgb.py
    ```

### 💡 Dicas e Solução de Problemas (Troubleshooting)

-   **Erro do XGBoost no macOS (`libomp.dylib not found`)**: Se você encontrar um erro ao treinar o XGBoost, instale a biblioteca OpenMP via Homebrew.
    ```bash
    brew install libomp
    ```
-   **Execução dos Scripts**: Sempre execute os scripts Python a partir do diretório raiz do projeto (`tcc-paysim/`) para que os caminhos relativos para as pastas `data/` e `models/` funcionem corretamente.

## 📂 Estrutura do Projeto
tcc-paysim/
├── data/              # Dados brutos e processados
├── models/            # Modelos de ML treinados e salvos
├── src/               # Código fonte principal
│   ├── pipelines/     # Scripts para o pipeline offline
│   └── streaming/     # Scripts para o pipeline de tempo real
├── venv/              # Ambiente virtual Python
├── .gitignore         # Arquivos a serem ignorados pelo Git
├── README.md          # Este arquivo
└── requirements.txt   # Dependências do projeto