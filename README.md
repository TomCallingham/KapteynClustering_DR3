# Module
Run "pip install -e ." in the KapteynClustering_DR3/KapteynClustering director to install the module.


## KapteynClustering Module
Install KapteynClustring by navigating to the folder and running:  
"pip install -e ."


# Paramaters
params are given in .yaml file. 


# Running From the Terminal
Run:  
"python Run_Script.py params.yaml" 

This will run pipeline for the steps in the steps array. For every step this would be:

[sample, artificial, linkage, significance, label]

These steps are also described in individual notebooks, linked below.


# NOTEBOOKS

## Initial Dataset and selections
[Initial Sample Data](./C1_Data.ipynb)

## Create Artifical DataSet
[Artificial Data ](./C2_Artificial_Data.ipynb)

## Single Linkage Cluster
[Single_Linkage](./C3_Single_Linkage.ipynb)

## Calculate Significance
[Find Significance](./C4_PCA_Analysis.ipynb)

## Significance Analysis
[Find how Signficant Clusters Are](./C5_Significance_Analysis.ipynb)

<!-- #region -->
# Running Kapetyn Clustering
Original implentation by Sofie Lövdal (described in Lövdal et al. (2021)).
Notebooks below take through the entire process
#  OLD
This notebook takes a Gaia-style catalogue and performs clustering in integral of motion space according to Lövdal et al. (2021).
Requires some libraries, most notably vaex, numpy and matplotlib. After changing the variable "catalogue_path" to indicate where your star catalogue is stored, you should be able to run the cells in order, and obtain a catalogue containing labels for significant clusters in integral of motion space.


The input catalogue should contain heliocentric xyz coordinates and xyz velocity components, where necessary quality cuts has been readily imposed. The input should contain halo stars and a little bit more, as we use vtoomre>180 when scrambling velocity components to obtain an artificial reference representation of a halo.

<!-- #endregion -->

```python

# KapteynClustering
## KapteynClustering Module

Install KapteynClustring by navigating to the folder and running:  
"pip install -e ."

# NOTEBOOKS

## Initial Dataset and selections
[Initial Sample Data](./C1_Data.ipynb)

## Create Artifical DataSet
[Artificial Data ](./C2_Artificial_Data.ipynb)

## Single Linkage Cluster
[Single_Linkage](./C3_Single_Linkage.ipynb)

## Calculate Significance
[Find Significance](./C4_PCA_Analysis.ipynb)

## Significance Analysis
[Find how Signficant Clusters Are](./C5_Significance_Analysis.ipynb)

# Repository
## Jupytext
Uses jupytext to save notebook conflict heartache.
Run .git_scripts/setup.sh to include a post merge hook that creates missing notebooks.
```
