# Documentação do Projeto PaySim - Detecção de Fraudes em Transações Financeiras

## Visão Geral
Este projeto utiliza Apache Spark e Kafka para realizar detecção de fraudes em transações financeiras simuladas pelo dataset PaySim. O pipeline abrange desde o processamento de dados, treinamento de modelos, interpretação dos resultados, até a inferência em tempo real via streaming.

## Estrutura dos Principais Arquivos

### 1. `1_data_scaling.py`
- **Função:** Realiza o carregamento do dataset completo, cria um subconjunto estratificado de 2 milhões de linhas e salva em formato Parquet.
- **Destaques:**
  - Amostragem estratificada para manter o balanceamento das classes.
  - Uso eficiente do Spark para grandes volumes de dados.
  - Salva o resultado em Parquet para uso posterior.

### 2. `2_offline_preprocessing_training.py`
- **Função:** Executa a engenharia de atributos, balanceamento das classes, pré-processamento e treinamento do modelo Random Forest com validação cruzada.
- **Destaques:**
  - Pipeline modular com etapas de indexação, codificação, normalização e montagem de vetores.
  - Balanceamento das classes via pesos.
  - Validação cruzada para seleção robusta do modelo.
  - Persistência do pipeline e do modelo treinado.

### 3. `3_model_interpretability.py`
- **Função:** Carrega o pipeline e o modelo treinado para analisar a importância dos atributos.
- **Destaques:**
  - Extração correta dos nomes dos atributos e suas importâncias.
  - Verificação de consistência entre nomes e importâncias.

### 4. `4_kafka_producer.py`
- **Função:** Realiza o streaming das transações do CSV para o tópico Kafka, simulando o fluxo de dados em tempo real.
- **Destaques:**
  - Conversão de tipos para garantir consistência.
  - Tratamento de erros em linhas inválidas.
  - Simulação de tempo real com intervalo entre envios.

### 5. `5_streaming_consumer.py`
- **Função:** Consome dados do Kafka, aplica o pipeline e o modelo para inferência em tempo real, e imprime alertas de fraude.
- **Destaques:**
  - Uso do Spark Structured Streaming para processamento contínuo.
  - Aplicação do pipeline e modelo persistidos.
  - Exibição dos alertas de fraude no console.

## Recomendações de Melhoria
- Adicionar tratamento de exceções e uso de logging para maior robustez.
- Parametrizar caminhos de arquivos e configurações via argumentos ou variáveis de ambiente.
- Expandir a documentação interna e comentários nos códigos.
- Considerar testes automatizados para funções críticas.
- Implementar visualização dos resultados (ex: gráficos de importância de atributos).

## Como Executar
1. **Preparação dos Dados:**
   - Execute `1_data_scaling.py` para gerar o subconjunto Parquet.
2. **Treinamento do Modelo:**
   - Execute `2_offline_preprocessing_training.py` para treinar e salvar o pipeline/modelo.
3. **Interpretação:**
   - Execute `3_model_interpretability.py` para visualizar a importância dos atributos.
4. **Streaming de Dados:**
   - Execute `4_kafka_producer.py` para enviar transações ao Kafka.
   - Execute `5_streaming_consumer.py` para consumir e detectar fraudes em tempo real.

## Dependências
- Python 3.x
- Apache Spark
- PySpark
- Kafka (servidor local)
- Biblioteca `kafka-python`

## Observações Finais
Este projeto está preparado para cenários de grande escala e pode ser expandido para produção com as melhorias sugeridas. Para dúvidas ou contribuições, consulte os comentários nos arquivos ou entre em contato com o responsável pelo projeto.
