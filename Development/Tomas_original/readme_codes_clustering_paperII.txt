
#######################
## Analysis of output from clustering algorithms (paper II):
#######################

In this folder you can find a set of notebooks and codes (and functions within) to reproduce all plots and analysis for paper II. Below some explanations on everything:

A) Catalogue creation: Basically, from the output from Sofie's code, this is used to add to original members all members within the Mahalanobis distance cut that we decided to use (80 percentile) --> Ask Sofie, she did this I think... I made an alternative code with the same results, but her code was nicer... probably you already have it.

B) Isochrone fitting: Isochrone fitting is done within codes, see below... but here a standalone code to run isochrone fitting given colour and magnitudes arrays.

Codes: reddening_computation.py, isochrone_fitting_modules.py, and clusters_iso_fit.ipynb.

C) FINAL_characterisation.ipynb

This notebook contains all plots from the analysis from D and E (see below)... basically all plots and tables that are shown in paper II, including appendixes and everything. We can go step by step so that you have a feeling.... some of this will be useful to you, most of it will change to your new analysis, data set, etc etc.... but here you have it just in case.

D) inspecting_dendrogram.ipynb

This is the basis of the analysis. This is a notebook to visualize the dendrogram from the clustering algorithm, you can see how to create the plots that we included in papers I and II (I am sure you can improve them!!). As I said, this is the starting point to everything. Here we tentatively define structures and substructures (A, A1, A2, C, C1, C2, C5, etc etc).... From these groupings of clusters we start the proper analysis in E. As I said, so far, although it is somehow automatise, it requires a lot of human input... I guess it can be made more automatic... but it won't be trivial... I hope all we did for these two papers is useful!

E) The analysis is divided into metallicity distribution functions (MDF) and colour distributions around a best fit isochrone (CaMD):

E.1) inspecting_substructures_DEF_3D_final.ipynb

E.2) inspecting_substructures_DEF_3D_CaMD_final.ipynb

Below I explain some of the main functions defined in each of the notebooks:

E.1.a) create_tables_defining: This is basically the function used to define A1, A2, A, C1, C3, etc etc... as well as to create tables like table 1 in paper 2. Basically it creates three types of tables:

   - definition_table: Original p-values from comparing each cluster with the union of the rest of the group where we want to check if it should be added or not.
   - % same decision... if we have a low number of stars, then we need to make sure that the KS p-value is reliable... for that, we do this (bootstrapping)... here I store the percentage of cases in which we end up having the same decision (above or below 0.05) as in the original (point above).
   - Less percentage. % of cases in bootstrapping in which pvalue<original p value.... in the ende this is not used.
   
In this function we create several plots. We use Nstar_lim  = 20 for doing all the bootstrapping and so on, i.e. if we have less than 20 stars, we start doing things.

E.1.b) create_table_adding_extra_members: This is the function to try to add new clusters to a group or structure that has been previously defined. With this function I create Table 2 in paper II. This is relatively similar to E.1.a, I also create 3 tables (definition, same decision and less percentage)... and it is use with a similar purpose.

E.1.c) Here we have 2 functions that do similar things. compare MDFs and compare MDFs low N stars. These are, I think, useful codes for making comparisons.

E.1.d) comp_2str_IoM: As the name indicates, this function compares in IoM the distribution of stars in two structures defined as groups of clusters following all this analysis.

E.1.e) nice_table_from_csv: This function is to transform hdf5 tables to latex format but also with the background colours, with characters of different colours as well, etc etc... --> I thought it added as well the percentages in parenthesis as in the tables in the paper, but it turns out that I added those manually, so I guess this is something you can improve.

 
 
All these functions are "repeated" with slight modifications in notebook E.2 (for that CaMD analysis)... those would be E.2.a to E.2.e... however, there are two that are unique to this notebook:

E.2.f) Apply_CaMD_fitting: In this case we are not comparing the MDF, but the distribution of colours around a best fitting isochrone.... so we need to apply a CaMD fitting that is done here. This function will call other functions that are in this notebook. The main output from here is to get the best isochrone to the structure under consideration as well as the distance of all stars to that isochrone so that the colour distributions can be compared via KS tests.

E.2.g) compare_distributions_CaMD: Basically for the final table comparing the distributions of colours with respect to the best isochrone for all structures identified.



F) For the Analysis that Eduardo is carrying out I also developed some codes to try to identify possible disrupted stars clusters.... this is done in the notebook GOOD_disrupted_star_clusters_March_2022.ipynb... I think this is better explained, but we can take a look at it together and explain everything that is done there... this is more automatic.... but this is only for the identification of possible disrupted clusters.... then Eduardo (or someone else) needs to do his analysis.... non-trivial...

