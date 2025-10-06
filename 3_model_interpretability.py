# File: 3_model_interpretability.py (Atualizado com Regressão Logística)
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel

def interpret_models():
    """
    Loads the saved models and pipeline to analyze feature importances (RF)
    and coefficients (LR).
    """
    spark = SparkSession.builder \
        .appName("ModelInterpretability") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    pipeline_path = 'models/preprocessing_pipeline_final'
    rf_model_path = 'models/best_rf_model_final'
    lr_model_path = 'models/best_lr_model_final'

    print("Loading saved pipeline and models...")
    pipeline_model = PipelineModel.load(pipeline_path)
    
    # --- Extrair Nomes das Features (comum a ambos os modelos) ---
    num_feature_names = pipeline_model.stages[2].getInputCols()
    string_indexer_model = pipeline_model.stages[0]
    cat_feature_labels = [f"type_{label}" for label in string_indexer_model.labels]
    feature_names = num_feature_names + cat_feature_labels


    # ## --- INTERPRETAÇÃO DO RANDOM FOREST --- ##
    print("\n" + "="*50)
    print("--- INTERPRETAÇÃO DO RANDOM FOREST ---")
    print("="*50)
    rf_model = RandomForestClassificationModel.load(rf_model_path)
    importances = rf_model.featureImportances.toArray()
    
    if len(feature_names) != len(importances):
        print("Error: Mismatch between feature names and RF importances.")
    else:
        rf_importance_map = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        print("\n--- Importância das Features (Random Forest) ---")
        for name, importance in rf_importance_map:
            print(f"{name:<25}: {importance:.4f}")


    # ## --- INTERPRETAÇÃO DA REGRESSÃO LOGÍSTICA --- ##
    print("\n" + "="*50)
    print("--- INTERPRETAÇÃO DA REGRESSÃO LOGÍSTICA ---")
    print("="*50)
    lr_model = LogisticRegressionModel.load(lr_model_path)
    coefficients = lr_model.coefficients.toArray()

    if len(feature_names) != len(coefficients):
        print("Error: Mismatch between feature names and LR coefficients.")
    else:
        lr_coeffs_map = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
        print("\n--- Coeficientes da Regressão Logística (do mais ao menos influente) ---")
        print("Um coeficiente POSITIVO aumenta a chance de fraude. Um NEGATIVO diminui.")
        for name, coeff in lr_coeffs_map:
            print(f"{name:<25}: {coeff:.4f}")

        
    spark.stop()

if __name__ == "__main__":
    interpret_models()