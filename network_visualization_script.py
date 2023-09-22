# Import libraries
from scipy import stats
from statsmodels.stats import multitest
import matplotlib.pyplot as plt
import pandas as pd
from graph_tool.all import *
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from graph_tool.all import minimize_nested_blockmodel_dl, minimize_blockmodel_dl

# Load data
file_path = 'data/Supplemental_Data_Raw_Genecounts.csv'
df = pd.read_csv(file_path)
df = df.drop('Unnamed: 17', axis=1)
df = df.rename(columns={"Unnamed: 0": "mnra"})
df.set_index('mnra', inplace=True)
df = df.apply(pd.to_numeric, errors='coerce', downcast='integer')
df = df[(df.loc[:, df.columns != 'index'] != 0).any(axis=1)]

# Calculating the mean and standard deviation for each row
df['mean'] = df.mean(axis=1)
df['std'] = df.std(axis=1)
df['cv'] = (df['std'] / df['mean']).abs() * 100

# Calculate significant_genes
group_a_cols = [col for col in df.columns if 'A' in col]
group_b_cols = [col for col in df.columns if 'B' in col]
results_df = pd.DataFrame(columns=['gene', 'mean_group_a', 'mean_group_b', 'p_value'])

for index, row in df.iterrows():
    group_a_data = row[group_a_cols]
    group_b_data = row[group_b_cols]
    mean_group_a = np.mean(group_a_data)
    mean_group_b = np.mean(group_b_data)
    t_stat, p_value = stats.ttest_ind(group_a_data, group_b_data, equal_var=False, nan_policy='omit')
    
    new_row = pd.DataFrame({
        'gene': [index],
        'mean_group_a': [mean_group_a],
        'mean_group_b': [mean_group_b],
        'p_value': [p_value]
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)
# Set 'gene' as index for results_df
results_df.set_index('gene', inplace=True)

# Add adjusted_p_value column
results_df['adjusted_p_value'] = multitest.multipletests(results_df['p_value'], method='fdr_bh')[1]

# Filter based on significance
significant_genes = results_df[results_df['adjusted_p_value'] < 0.01]
print(significant_genes)
# Calculate common_genes
top_cv_genes_df = df.nlargest(significant_genes.shape[0], 'cv')
print("First few index values of significant_genes:", significant_genes.index[:5])
print("First few index values of top_cv_genes_df:", top_cv_genes_df.index[:5])
common_genes = set(significant_genes.index).intersection(set(top_cv_genes_df.index))

# Standardize the data
print("Shape of significant_genes:", significant_genes.shape)
print("Shape of top_cv_genes_df:", top_cv_genes_df.shape)
common_genes_df = df.loc[list(common_genes)]
scaler = StandardScaler()
# Add these debugging lines here
print("Shape of common_genes_df:", common_genes_df.shape)
print("First rows of common_genes_df:", common_genes_df.head())
standardized_data = scaler.fit_transform(common_genes_df.T)

# Initialize the graph
g = Graph(directed=False)
v_label = g.new_vertex_property("string")
gene_to_vertex = {}

expression_data = common_genes_df.drop(columns=['mean', 'std', 'cv'])
correlation_matrix = expression_data.T.corr()

# Create vertices and edges in the graph
for gene in expression_data.index:
    v = g.add_vertex()
    v_label[v] = gene
    gene_to_vertex[gene] = v

edge_weight = g.new_edge_property("double")
for i, gene1 in enumerate(expression_data.index):
    for j, gene2 in enumerate(expression_data.index):
        if i >= j:
            continue
        correlation = correlation_matrix.loc[gene1, gene2]
        if correlation > 0.6:
            v1 = gene_to_vertex[gene1]
            v2 = gene_to_vertex[gene2]
            e = g.add_edge(v1, v2)
            edge_weight[e] = np.abs(correlation) * 3

# Save graph properties
g.vertex_properties["label"] = v_label
g.edge_properties["weight"] = edge_weight


state = minimize_nested_blockmodel_dl(g)

state.draw(output="power_nested_mdl.svg")
