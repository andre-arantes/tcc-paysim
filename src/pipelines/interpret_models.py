# src/pipelines/6_interpret_models.py

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel

def interpret_lr_model():
    """
    Carrega o pipeline e o modelo de Regressão Logística salvo para analisar
    os coeficientes das features.
    """
    spark = SparkSession.builder \
        .appName("InterpretLogisticRegression") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    # Caminhos para o pipeline e para o modelo de Regressão Logística
    pipeline_path = 'models/preprocessing_pipeline_final'
    lr_model_path = 'models/best_lr_model_final'

    print("Loading saved pipeline...")
    pipeline_model = PipelineModel.load(pipeline_path)
    
    # --- Extrair Nomes das Features (comum a todos os modelos) ---
    num_feature_names = pipeline_model.stages[2].getInputCols()
    string_indexer_model = pipeline_model.stages[0]
    cat_feature_labels = [f"type_{label}" for label in string_indexer_model.labels]
    feature_names = num_feature_names + cat_feature_labels
    print(f"Total de features para análise: {len(feature_names)}")


    # ## --- INTERPRETAÇÃO DA REGRESSÃO LOGÍSTICA --- ##
    print("\n" + "="*50)
    print("--- INTERPRETAÇÃO DA REGRESSÃO LOGÍSTICA ---")
    print("="*50)
    try:
        lr_model = LogisticRegressionModel.load(lr_model_path)
        coefficients = lr_model.coefficients.toArray()

        if len(feature_names) == len(coefficients):
            lr_coeffs_map = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
            print("\n--- Coeficientes da Regressão Logística (do mais ao menos influente) ---")
            print("Um coeficiente POSITIVO aumenta a chance de fraude. Um NEGATIVO diminui.")
            for name, coeff in lr_coeffs_map:
                print(f"{name:<25}: {coeff:.4f}")
        else:
            print("Erro: Incompatibilidade entre número de features e coeficientes.")
    except Exception as e:
        print(f"Não foi possível carregar ou interpretar o modelo de Regressão Logística. Erro: {e}")
        
    spark.stop()

if __name__ == "__main__":
    interpret_lr_model()