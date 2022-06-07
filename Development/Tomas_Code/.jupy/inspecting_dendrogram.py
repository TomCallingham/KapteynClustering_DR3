# %%
import numpy as np
import vaex
import vaex.ml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram
from KapteynClustering import grouping_funcs as groupf

# %% [markdown]
# # Now dendrogram in 3D, avoiding circularity!!!
#
# # Mahalanobis distance in 3D (probability from CDF) and re-doing everything from scratch!

# %%
# df = vaex.open('/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/df_labels_3D_3sigma.hdf5')
df = vaex.open(
    '/net/gaia2/data/users/callingham/data/clustering/KapteynClustering_DR3/Development/emma_copy_df_labels_3D_3sigma.hdf5')
df
# %%
from KapteynClustering import param_funcs as paramf
params = paramf.read_param_file("../../Params/gaia_params.yaml")
features = params["cluster"]["features"]

# %%

# %%
# BAD group. covariance high?
df.select('(labels==67)', name='bad')
plt.figure()
plt.hist(df.evaluate('dec_parallax_corr', selection='bad'), histtype='step', color='blue')
plt.hist(df.evaluate('parallax_pmdec_corr', selection='bad'), histtype='step', color='red')
plt.show()
# ((df.evaluate('dec_parallax_corr')<-0.15)|(df.evaluate('parallax_pmdec_corr')>0.15))

# %%
labels_array = np.array(df['labels'].values)
maxsig_array = np.array(df['maxsig'].values)
labels_array[(labels_array == 67)] = 0.0
maxsig_array[(labels_array == 67)] = 0.0

Groups = np.unique(labels_array[maxsig_array > 3])

df.add_column('labels', labels_array)
df.add_column('maxsig', maxsig_array)

# %%

# %%

# %%
dm = groupf.get_cluster_distance_matrix(df[df.maxsig>3], features=features,distance_metric="mahalanobis")
dm = groupf.get_cluster_distance_matrix(df[df.maxsig>3], features=features,distance_metric="euclidean")


# %%
def plot_cluster_relation(df, distance_metric):

    # Customize a distance matrix, as if each cluster would be a single data point.
    # You can either measure the distance between the clusters by their 'mahalanobis' distance
    # or 'euclidean' distance.
    dm = groupf.get_cluster_distance_matrix(df, features, Groups, distance_metric)[0]

    # Perform single linkage with the custom made distance matrix
    Z = linkage(dm, 'single')

    def llf(id):
        '''For labelling the leaves in the dendrogram according to cluster index'''
        return str(Groups[id])

    plt.subplots(figsize=(20, 5))
    dendrogram(Z, leaf_label_func=llf, leaf_rotation=90, color_threshold=6)
    plt.title(f'Relationship between 3$\\sigma$ clusters (single linkage by = {distance_metric} distance)')

    return Z, llf



# %%
Z_mahalanobis, labels_mahalanobis = plot_cluster_relation(df[df.maxsig > 3], 'mahalanobis')

# %%
Z_euclidean, labels_euclidean = plot_cluster_relation(df[df.maxsig > 3], 'euclidean')

# %%
# Option B:

N_struct = 6
x_pos = [74., 163.5, 200.5, 572.0, 640, 660]
y_pos = [4.3, 4.3, 4.3, 4.3, 4.3, 4.3]
x_label = ['A', 'B', 'C', 'D', 'E', 'F']

cmap = cm.jet(np.linspace(0, 1, N_struct))
cmap = ['#01befe', '#ffdd00', '#ff7d00', '#ff006d', '#adff02', '#8f00ff']
hierarchy.set_link_color_palette([rgb for rgb in cmap])

fig, ax = plt.subplots(1, 1, figsize=(30, 6), facecolor='white')
asdf = dendrogram(Z_mahalanobis, leaf_label_func=labels_mahalanobis,
                  color_threshold=4.0, above_threshold_color='grey')
ax.tick_params(labelsize=15)
ax.axhline(y=4.0, c='grey', lw=1, linestyle='dashed')
#plt.title(f'Relationship between 3$\sigma$ clusters (single linkage by mahalanobis distance)')
for i in np.arange(N_struct):
    ax.text(x_pos[i], y_pos[i], x_label[i], size=16, color=cmap[i], ha="center", va="center",
            bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)))  # , weight='bold')

#ax.text(604, 6.5, 'E', size=16, color='grey', ha="center", va="center", bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)), weight='bold')


ax.set_ylabel(r'$\rm Mahalanobis \; distance$', fontsize=22)

plt.xticks(rotation=45)

plt.ylim(0.0, 10.5)

plt.show()
# mahalanobis_distance

# %%
print(asdf['ivl'])

# %%
# Option B + Gaia-Enceladus:

N_struct = 6
x_pos = [74., 163.5, 200.5, 572.0, 640, 660]
y_pos = [4.3, 4.3, 4.3, 4.3, 4.3, 4.3]
x_label = ['A', 'B', 'C', 'D', 'E', 'F']

cmap = cm.jet(np.linspace(0, 1, N_struct))
cmap = ['#01befe', '#ffdd00', '#ff7d00', '#ff006d', '#adff02', '#8f00ff']
hierarchy.set_link_color_palette([rgb for rgb in cmap])

fig, ax = plt.subplots(1, 1, figsize=(30, 6), facecolor='white')
asdf = dendrogram(Z_mahalanobis, leaf_label_func=labels_mahalanobis,
                  color_threshold=4.0, above_threshold_color='grey')
ax.tick_params(labelsize=15)
ax.axhline(y=4.0, c='grey', lw=1, linestyle='dashed')
#plt.title(f'Relationship between 3$\sigma$ clusters (single linkage by mahalanobis distance)')
for i in np.arange(N_struct):
    ax.text(x_pos[i], y_pos[i], x_label[i], size=16, color=cmap[i], ha="center", va="center",
            bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)))  # , weight='bold')

#ax.text(604, 6.5, 'E', size=16, color='grey', ha="center", va="center", bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)), weight='bold')

# 1:
x_pos = [53.5, 80]
y_pos = [3.4, 3.55] / np.sqrt(2)
x_label = ['A1', 'A2']
for i in np.arange(len(x_label)):
    ax.text(x_pos[i], y_pos[i], x_label[i], size=16, color=cmap[0], ha="center", va="center", bbox=dict(
        boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)))  # , weight='bold', alpha=0.8)

# Gaia-Enceladus labels:
x_pos = [200.5, 243, 304, 385, 502, 543]
y_pos = [4.5, 4.55, 4.55, 4.55, 3.7, 3.8] / np.sqrt(2)
x_label = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
for i in np.arange(len(x_label)):
    ax.text(x_pos[i], y_pos[i], x_label[i], size=16, color=cmap[2], ha="center", va="center", bbox=dict(
        boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)))  # , weight='bold', alpha=0.8)

ax.set_ylabel(r'$\rm Mahalanobis \; distance$', fontsize=22)

plt.xticks(rotation=45)

plt.ylim(0.0, 10.5)

plt.show()

# %%
print(asdf['ivl'])

# %%
grouped_clusters = [
    [ 33, 12, 40, 34, 28, 32, 31, 15, 16],
    [ 42, 46, 38, 18, 30],
    [ 37, 8, 11],
    [ 6, 56, 59, 52, 14, 10, 7, 13, 20, 19, 57, 58, 54, 55, 50, 53, 47, 44, 51, 48, 45, 25, 29, 41, 49, 17, 43, 39, 36, 23, 5, 27, 22, 26, 21, 35, 9, 24],
    [ 64, 62, 63],
    [ 2, 1, 3],
    [ 60, 61],
    [ 65, 66]]
independent_clusters = [68, 4]
possible_independent_clusters = [14, ]
