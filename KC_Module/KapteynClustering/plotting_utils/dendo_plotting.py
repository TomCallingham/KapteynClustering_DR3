from KapteynClustering import linkage_funcs as linkf
from KapteynClustering.plotting_utils import plotting_scripts as ps
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

def dendo_plot(Z, Grouped_Clusters, Groups, Clusters, threshold_cut=None, G_cols=None, C_cols=None):
    tree = linkf.find_tree(Z,prune=True)
    Nlinks = len(Z)
    Nclusters = len(Clusters) #Nlinks+1
    if G_cols is None:
        G_cols = ps.get_G_colours(Groups)
    if C_cols is None:
        C_cols = ps.get_G_colours(np.arange(len(Grouped_Clusters)))

    dendo_colours = np.full((Nlinks+1),"k", dtype = "U32")
    for n in range(Nlinks):
        members = tree[n+Nlinks]
        m0 = members[0]
        G = Grouped_Clusters[m0]
        Pop_check = (Grouped_Clusters==G).sum()
        if len(members)<=Pop_check:
            dendo_colours[n] = G_cols.get(G,"lightgrey")

    def get_dendo_colours(k):
        try:
            c = dendo_colours[k-(Nlinks)]
        except Exception as e:
            c = "k"
        return c
    def llf(id):
        '''For labelling the leaves in the dendrogram according to cluster index'''
        return str(Clusters[id])

    plt.subplots(figsize=(20,10))
    d_result = dendrogram(Z, leaf_rotation=90,  link_color_func=get_dendo_colours,leaf_label_func=llf)
    if threshold_cut is not None:
        plt.axhline(threshold_cut, c="r")

    xbase, ybase, leaves = find_dendo_base_xy(d_result)

    Group_Pops = {g:(Grouped_Clusters==g).sum() for g in Groups}
    for (l,x,y) in zip(leaves,xbase,ybase):
        c = C_cols[l]
        # plt.scatter([x],[y/2],c=c)
        plt.scatter([x],[1],c=c,zorder=10)
        # plt.text(x,y,l)
        if Group_Pops[Grouped_Clusters[l]]==1:
            plt.plot([x,x],[0,y],c=c)
    plt.minorticks_off()
    return

def find_dendo_base_xy(d_result):
    leaves = d_result["leaves"]
    icoord = np.array(d_result["icoord"])
    dcoord = np.array(d_result["dcoord"])
    xstick = np.concatenate((icoord[:,:2],icoord[:,2:]))
    ystick = np.concatenate((dcoord[:,:2],dcoord[:,2:][:,::-1]))
    base = ystick[:,0] ==0
    base_xstick = xstick[base,0]
    base_ystick = ystick[base,1]
    ix_sort = np.argsort(base_xstick)
    xbase,ybase = base_xstick[ix_sort], base_ystick[ix_sort]
    return xbase, ybase, leaves
