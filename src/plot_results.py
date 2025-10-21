import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


# Dados da Regressão Logística
lr_results = {
    'threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'precision': [0.9000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    'recall':    [0.0353, 0.0353, 0.0314, 0.0196, 0.0118, 0.0039, 0.0039],
    'f1_score':  [0.0679, 0.0682, 0.0608, 0.0385, 0.0233, 0.0078, 0.0078]
}

# Dados do Random Forest
rf_results = {
    'threshold': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    'precision': [0.7876, 0.8191, 0.8530, 0.8764, 0.9277, 0.9536, 0.9640], 
    'recall':    [0.9060, 0.9023, 0.8947, 0.8797, 0.8684, 0.8496, 0.8045],
    'f1_score':  [0.8427, 0.8587, 0.8734, 0.8780, 0.8971, 0.8986, 0.8770]
}

# Dados do XGBoost
xgb_results = {
    'threshold': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    'precision': [0.7876, 0.8191, 0.8530, 0.8764, 0.9277, 0.9536, 0.9640],
    'recall':    [0.9060, 0.9023, 0.8947, 0.8797, 0.8684, 0.8496, 0.8045],
    'f1_score':  [0.8427, 0.8587, 0.8734, 0.8780, 0.8971, 0.8986, 0.8770]
}

# Identifique os melhores F1-Scores e thresholds manualmente a partir dos logs
best_thresholds = {
    'LR': 0.50,
    'RF': 0.50,
    'XGB': 0.95 # Este foi o melhor F1 para XGBoost
}

final_metrics_xgb_095 = {
    'AUC': 0.9991, # Use a AUC correta do seu último run do evaluate_final
    'Precisão': 0.9615,
    'Recall': 0.8032,
    'F1-Score': 0.8753
}

# --- Criação dos DataFrames Pandas ---
lr_df = pd.DataFrame(lr_results)
rf_df = pd.DataFrame(rf_results)
xgb_df = pd.DataFrame(xgb_results)

# Adiciona coluna com nome do modelo para facilitar plotagem
lr_df['Modelo'] = 'Regressão Logística'
rf_df['Modelo'] = 'Random Forest'
xgb_df['Modelo'] = 'XGBoost'

# Combina todos os resultados
all_results_df = pd.concat([lr_df, rf_df, xgb_df], ignore_index=True)

# Define estilo dos gráficos
sns.set_theme(style="whitegrid")

# --- Gráfico 1: Curva Precisão x Recall ---
plt.figure(figsize=(10, 7))
sns.lineplot(data=all_results_df, x='recall', y='precision', hue='Modelo', marker='o', ci=None)

# Anota os pontos de melhor F1-Score
for model_name, threshold in best_thresholds.items():
    if model_name == 'LR': df = lr_df
    elif model_name == 'RF': df = rf_df
    else: df = xgb_df
    
    best_point = df[df['threshold'] == threshold].iloc[0]
    plt.scatter(best_point['recall'], best_point['precision'], s=150, facecolors='none', edgecolors='red', linewidth=2)
    plt.text(best_point['recall'] + 0.01, best_point['precision'], f"Best F1 ({model_name})\nT={threshold:.2f}", fontsize=9, color='red')

plt.title('Curva Precisão x Recall por Modelo (Thresholds Variados)')
plt.xlabel('Recall (Sensibilidade)')
plt.ylabel('Precisão')
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.legend(title='Modelo')
plt.grid(True)
plt.tight_layout()
# Salva o gráfico
plot_filename_pr = "grafico_precisao_recall.png"
plt.savefig(plot_filename_pr)
print(f"Gráfico Precisão x Recall salvo como '{plot_filename_pr}'")
# plt.show() # Descomente para exibir interativamente

# --- Gráfico 2: F1-Score x Threshold ---
plt.figure(figsize=(10, 7))
sns.lineplot(data=all_results_df, x='threshold', y='f1_score', hue='Modelo', marker='o', ci=None)

# Anota os pontos de melhor F1-Score
for model_name, threshold in best_thresholds.items():
    if model_name == 'LR': df = lr_df
    elif model_name == 'RF': df = rf_df
    else: df = xgb_df
    
    best_point = df[df['threshold'] == threshold].iloc[0]
    plt.scatter(best_point['threshold'], best_point['f1_score'], s=150, facecolors='none', edgecolors='red', linewidth=2)
    plt.text(best_point['threshold'] + 0.01, best_point['f1_score'], f"Max F1\n{best_point['f1_score']:.3f}", fontsize=9, color='red')

plt.title('F1-Score x Threshold de Decisão por Modelo')
plt.xlabel('Threshold de Decisão (Probabilidade de Fraude)')
plt.ylabel('F1-Score')
plt.ylim(0, 1.0)
plt.legend(title='Modelo')
plt.grid(True)
plt.tight_layout()
# Salva o gráfico
plot_filename_f1 = "grafico_f1_threshold.png"
plt.savefig(plot_filename_f1)
print(f"Gráfico F1-Score x Threshold salvo como '{plot_filename_f1}'")
# plt.show() # Descomente para exibir interativamente

# --- Gráfico 3: Métricas Finais do Melhor Modelo ---
plt.figure(figsize=(8, 5))
final_metrics_names = list(final_metrics_xgb_095.keys())
final_metrics_values = list(final_metrics_xgb_095.values())

sns.barplot(x=final_metrics_names, y=final_metrics_values, palette="viridis")
plt.title(f'Métricas Finais - XGBoost (Threshold = {best_thresholds["XGB"]:.2f}) no Conjunto de Teste')
plt.ylabel('Score')
plt.ylim(0, 1.05)
# Adiciona os valores no topo das barras
for index, value in enumerate(final_metrics_values):
    plt.text(index, value + 0.01, f"{value:.4f}", ha='center', fontsize=10)

plt.tight_layout()
# Salva o gráfico
plot_filename_final = "grafico_metricas_finais_xgb.png"
plt.savefig(plot_filename_final)
print(f"Gráfico de Métricas Finais salvo como '{plot_filename_final}'")
# plt.show() # Descomente para exibir interativamente

print("\nGráficos gerados com sucesso!")

# Cria a pasta 'plots' se não existir
# output_dir = "plots"
# os.makedirs(output_dir, exist_ok=True)
# plot_filename_pr = os.path.join(output_dir, "grafico_precisao_recall.png")
# plot_filename_f1 = os.path.join(output_dir, "grafico_f1_threshold.png")
# plot_filename_final = os.path.join(output_dir, "grafico_metricas_finais_xgb.png")