import pandas as pd
import matplotlib.pyplot as plt
import io

# 1. Données
data = [
    {"tested_var": "guidance_scale", "tested_value": 1.0, "Avg_Emotion_Score": 0.765419, "Avg_Face_Prob": 0.998792},
    {"tested_var": "guidance_scale", "tested_value": 2.0, "Avg_Emotion_Score": 0.791970, "Avg_Face_Prob": 0.999408},
    {"tested_var": "guidance_scale", "tested_value": 5.0, "Avg_Emotion_Score": 0.845778, "Avg_Face_Prob": 0.998286},
    {"tested_var": "guidance_scale", "tested_value": 10.0, "Avg_Emotion_Score": 0.876170, "Avg_Face_Prob": 0.999158},
    
    {"tested_var": "noise_strength", "tested_value": 0.1, "Avg_Emotion_Score": 0.745735, "Avg_Face_Prob": 0.999359},
    {"tested_var": "noise_strength", "tested_value": 0.2, "Avg_Emotion_Score": 0.884936, "Avg_Face_Prob": 0.998141},
    {"tested_var": "noise_strength", "tested_value": 0.3, "Avg_Emotion_Score": 0.918425, "Avg_Face_Prob": 0.997084},
    {"tested_var": "noise_strength", "tested_value": 0.7, "Avg_Emotion_Score": 0.956443, "Avg_Face_Prob": 0.999568},
    
    {"tested_var": "steps", "tested_value": 50.0, "Avg_Emotion_Score": 0.778619, "Avg_Face_Prob": 0.999134},
    {"tested_var": "steps", "tested_value": 150.0, "Avg_Emotion_Score": 0.831793, "Avg_Face_Prob": 0.998514},
    {"tested_var": "steps", "tested_value": 250.0, "Avg_Emotion_Score": 0.878708, "Avg_Face_Prob": 0.999445},
    {"tested_var": "steps", "tested_value": 500.0, "Avg_Emotion_Score": 0.875816, "Avg_Face_Prob": 0.999027},
]

df = pd.DataFrame(data)

# 2. Création des graphiques
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
variables = ['guidance_scale', 'noise_strength', 'steps']
titles = ['Guidance Scale', 'Noise Strength', 'Steps']

for i, var in enumerate(variables):
    subset = df[df['tested_var'] == var]
    ax = axes[i]
    
    # X: Valeur testée (traitée comme catégorie)
    x_labels = subset['tested_value'].astype(str).tolist()
    x_pos = range(len(x_labels))
    
    # Y: Score émotion
    y = subset['Avg_Emotion_Score']
    # Taille: Probabilité visage (multipliée pour être visible)
    s = subset['Avg_Face_Prob'] * 300 
    
    # Scatter plot sur positions catégorielles
    scatter = ax.scatter(x_pos, y, s=s, alpha=0.7, c='blue', edgecolors='black')
    
    # Configuration des axes
    ax.set_ylim(0, 1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    
    
    ax.set_xlabel(titles[i])
    ax.set_ylabel("Avg Emotion Score")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Annoter les points avec la proba visage pour plus de clarté
    for j, (idx, row) in enumerate(subset.iterrows()):
        ax.annotate(f"{row['Avg_Face_Prob']:.4f}", 
                    (j, row['Avg_Emotion_Score']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=8)

plt.tight_layout()
plt.savefig('benchmark_graphs.png')
print("Graphiques sauvegardés dans 'benchmark_graphs.png'")
plt.show()
