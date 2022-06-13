import sys
import os

from KapteynClustering.RunSteps import c7_cluster_fits
try:
    from .RunSteps import c1_sample_prepare, c2_art_data, c3_single_linkage, c4_significance_find, c5_find_clusters, c7_cluster_fits, c8_cluster_dendogram, c9_grouping_clusters
    import param_funcs as paraf
except Exception:
    from KapteynClustering.RunSteps import c1_sample_prepare, c2_art_data, c3_single_linkage, c4_significance_find, c5_find_clusters, c7_cluster_fits, c8_cluster_dendogram, c9_grouping_clusters
    import KapteynClustering.param_funcs as paraf


def main(param_file):
    print("Running KC script, using params found:")
    print(param_file, "\n")
    params = paraf.read_param_file(param_file)

    if "multiple" in params.keys():
        multiple_main(param_file, params)
        return

    steps = params["run"]["steps"]
    print("Applying following steps:")
    print(steps, '\n')
    if "sample" in steps:
        c1_sample_prepare.sample_prepare(params)
    if "artificial" in steps:
        c2_art_data.art_data(params)
    if "linkage" in steps:
        c3_single_linkage.single_linkage(params)
    if "significance" in steps:
        c4_significance_find.significance_find(param_file)
    if "clusters" in steps:
        c5_find_clusters.find_clusters(params)
    if "cluster_fits" in steps:
        c7_cluster_fits.cluster_fit_step(params)
    if "cluster_dendogram" in steps:
        c8_cluster_dendogram.cluster_dendogram(params)
    if "group_feh" in steps:
        c9_grouping_clusters.group_feh(params)
    return


def multiple_main(param_file, params):
    print("Multiple Run!")
    multi_stage = params["multiple"]["stage"]
    print(f"multiple from stage: {multi_stage}")
    multi_list = paraf.list_files(params["data"][multi_stage])
    n_multi_current = params["multiple"].get("current", 0)
    N_samples = len(multi_list)
    print("Looping over files")
    for n_multi in range(n_multi_current, N_samples):
        print("creating multi file:")
        multi_param_file = paraf.multi_create_param_file(
            param_file, n_multi=n_multi)
        print("running main on file")
        main(multi_param_file)
        os.remove(multi_param_file)
    return


if __name__ == "__main__":
    param_file = sys.argv[1]
    main(param_file)
