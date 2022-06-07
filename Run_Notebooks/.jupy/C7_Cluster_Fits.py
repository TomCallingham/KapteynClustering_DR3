# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.label_funcs as labelf
import KapteynClustering.cluster_funcs as clusterf
import KapteynClustering.mahalanobis_funcs as mahaf

# %%
from KapteynClustering.plot_funcs import plot_simple_scatter
from KapteynClustering.plotting_utils import plotting_scripts as ps 

# %% [markdown]
# ### Params

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]
min_sig=params["label"]["min_sig"]
features = params["cluster"]["features"]

# %% [markdown]
# # Load

# %%
stars = vaex.open(data_p["labelled_sample"])

label_data = dataf.read_data(fname= data_p["label"])
[labels, fGroups, fPops, fG_sig] = [ label_data[p] for p in
                                           ["labels", "Groups", "Pops", "G_sig"]]

# %%
Groups, Pops, G_sig = fGroups[fGroups!=-1], fPops[fGroups!=-1], fG_sig[fGroups!=-1]

# %% [markdown]
# # Make Fits

# %%
print(len(Groups))

# %%
X = clusterf.find_X(features, stars, scaled=False)

# %%
mean_group = {}
cov_group = {}
# print("Needs bias correct!")
for g in Groups:
    g_filt = (labels == g)
    # mean_group[g], cov_group[g] = mahaf.fit_gaussian_bias(X[g_filt, :])
    mean_group[g] , cov_group[g] = mahaf.fit_gaussian(X[g_filt,:])

# %%
fit_data = {"Groups":Groups, "Pops":Pops, "G_sig":G_sig,
            "mean":mean_group, "covariance:":cov_group,
            "features":features}
dataf.write_data(fname=data_p["gaussian_fits"], dic=fit_data)


# %% [markdown]
# # Adding Stars To Clusters

# %%
def cluster_maha_dis(stars, Clusters, cluster_mean, cluster_cov)
    return
def add_members(stars, fit_file = data_p["gaussian_fits"], m_dis=2):
    dis = cluster_maha_dis(stars, fite_file)
    return


# %% [markdown]
# # Plots

# %%
G_Colours = ps.get_G_colours(Groups)
G_Colours_list = np.array([G_Colours[g] for g in Groups])

# %%
# To plot use Similar code to GC project on cosma!
