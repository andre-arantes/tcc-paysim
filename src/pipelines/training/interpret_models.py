# src/pipelines/6_interpret_models.py

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel
from xgboost.spark import SparkXGBClassifierModel

def interpret_models():
    """
    Carrega os modelos salvos e o pipeline para analisar a importância das features (RF, XGBoost)
    e os coeficientes (LR).
    """
    spark = SparkSession.builder \
        .appName("ModelInterpretability") \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .config("spark.driver.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    # Caminhos atualizados relativos à raiz do projeto
    pipeline_path = 'models/preprocessing_pipeline_final'
    rf_model_path = 'models/best_rf_model_weighted'      
    lr_model_path = 'models/best_lr_model_final'      
    xgb_model_path = 'models/best_xgb_model_final'    
    
    print("Loading saved pipeline...")
    pipeline_model = PipelineModel.load(pipeline_path)
    
    # --- Extrair Nomes das Features (comum a todos os modelos) ---
    num_feature_names = pipeline_model.stages[2].getInputCols()
    string_indexer_model = pipeline_model.stages[0]
    # O .labels do StringIndexer contém as categorias ('PAYMENT', 'CASH_OUT', etc.)
    cat_feature_labels = [f"type_{label}" for label in string_indexer_model.labels]
    feature_names = num_feature_names + cat_feature_labels
    print(f"Total de features para análise: {len(feature_names)}")


    # ## --- INTERPRETAÇÃO DO RANDOM FOREST --- ##
    print("\n" + "="*50)
    print("--- INTERPRETAÇÃO DO RANDOM FOREST ---")
    print("="*50)
    try:
        rf_model = RandomForestClassificationModel.load(rf_model_path)
        importances = rf_model.featureImportances.toArray()
        
        if len(feature_names) == len(importances):
            rf_importance_map = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            print("\n--- Importância das Features (Random Forest) ---")
            for name, importance in rf_importance_map:
                print(f"{name:<25}: {importance:.4f}")
        else:
            print(f"Erro: Incompatibilidade! Nomes de features: {len(feature_names)}, Importâncias: {len(importances)}")
    except Exception as e:
        print(f"Não foi possível carregar ou interpretar o modelo Random Forest. Erro: {e}")


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
            print(f"Erro: Incompatibilidade! Nomes de features: {len(feature_names)}, Coeficientes: {len(coefficients)}")
    except Exception as e:
        print(f"Não foi possível carregar ou interpretar o modelo de Regressão Logística. Erro: {e}")

        
    # ## --- INTERPRETAÇÃO DO XGBOOST --- ##
    print("\n" + "="*50)
    print("--- INTERPRETAÇÃO DO XGBOOST ---")
    print("="*50)
    try:
        xgb_model = SparkXGBClassifierModel.load(xgb_model_path)
        feature_importances_dict = xgb_model.get_feature_importances()
        
        xgb_importance_map = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)

        print("\n--- Importância das Features (XGBoost) ---")
        for f_index, importance in xgb_importance_map:
            feature_index = int(f_index.replace('f', ''))
            if feature_index < len(feature_names):
                feature_name = feature_names[feature_index]
                print(f"{feature_name:<25}: {importance:.4f}")
    except Exception as e:
        print(f"Não foi possível carregar ou interpretar o modelo XGBoost. Erro: {e}")

        
    spark.stop()

if __name__ == "__main__":
    interpret_models()