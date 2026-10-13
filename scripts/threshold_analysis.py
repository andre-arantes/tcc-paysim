import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for script
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Data from model evaluation
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
precision = [0.6963, 0.7323, 0.7678, 0.7937, 0.8527, 0.8944, 0.9615]
recall = [0.9438, 0.9337, 0.9297, 0.9036, 0.8835, 0.8675, 0.8032]
f1_score = [0.8014, 0.8208, 0.8411, 0.8451, 0.8679, 0.8807, 0.8753]

# Create plot with larger size and clearer fonts
fig, ax = plt.subplots(figsize=(10, 6))

# Improved line styles and larger markers for readability
ax.plot(thresholds, precision, marker='o', color='tab:blue', label='Precisão', linewidth=2, markersize=8)
ax.plot(thresholds, recall, marker='o', color='tab:green', label='Recall', linewidth=2, markersize=8)
ax.plot(thresholds, f1_score, marker='o', color='tab:red', label='F1-Score', linewidth=2.2, markersize=9)

# Labels and title (Portuguese as in file)
ax.set_xlabel('Limiar de Decisão', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precisão, Recall, e F1-Score vs. Limiar de Decisão', fontsize=14, weight='semibold')
ax.set_ylim(0.0, 1.05)

# Major/minor grid and ticks for better readability
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.grid(which='major', linestyle='--', alpha=0.6)
ax.grid(which='minor', linestyle=':', alpha=0.25)

ax.legend(fontsize=11)

# Visual highlights — clearer vertical lines and markers
max_f1 = max(f1_score)
max_f1_idx = f1_score.index(max_f1)
max_f1_threshold = thresholds[max_f1_idx]
# Strong dashed line for optimal F1
ax.axvline(x=max_f1_threshold, color='tab:red', linestyle='--', linewidth=2, alpha=0.9)
# Emphasize the exact point
ax.plot([max_f1_threshold], [max_f1], marker='D', color='tab:red', markersize=10)
ax.annotate(f'Max F1 = {max_f1:.4f}\nLimiar = {max_f1_threshold}',
            xy=(max_f1_threshold, max_f1), xycoords='data',
            xytext=(15, -40), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='tab:red'))

# Selected business threshold (dotted line) — more visible styling
business_threshold = 0.99
ax.axvline(x=business_threshold, color='black', linestyle=':', linewidth=2)
ax.plot([business_threshold], [f1_score[-1]], marker='X', color='black', markersize=10)
ax.annotate(f'Limiar selecionado = {business_threshold}',
            xy=(business_threshold, f1_score[-1]), xycoords='data',
            xytext=(-140, 20), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='black'))

# Tight layout and save higher-resolution figure
output_path = 'reports/figures/threshold_analysis.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f'Saved plot to {output_path}')
