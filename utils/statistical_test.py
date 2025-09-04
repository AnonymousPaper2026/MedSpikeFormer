import numpy as np
import pandas as pd
from scipy import stats

data={}

df = pd.DataFrame(data)

our_row = df[df['Model'] == 'Our'].iloc[0]
other_models = df[df['Model'] != 'Our']

results = []

for _, row in other_models.iterrows():
    model_name = row['Model']
    our_scores = our_row[1:].values.astype(float)
    model_scores = row[1:].values.astype(float)
    
    t_stat, p_value = stats.ttest_rel(our_scores, model_scores)
    results.append({'Model vs. Ours': model_name, 'p-value': p_value})

results_df = pd.DataFrame(results)

results_df['p-value'] = results_df['p-value'].apply(lambda x: f"{x:.4f}")

print(results_df.to_markdown(index=False))