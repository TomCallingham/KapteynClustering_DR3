import sys
import os
try:
    from .RunSteps import sample_prepare, art_data, single_linkage, significance_find, find_labels
    import param_funcs as paraf
except Exception:
    from KapteynClustering.RunSteps import sample_prepare, art_data, single_linkage, significance_find, find_labels
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
        sample_prepare.sample_prepare(params)
    if "artificial" in steps:
        art_data.art_data(params)
    if "linkage" in steps:
        single_linkage.single_linkage(params)
    if "significance" in steps:
        significance_find.significance_find(param_file)
    if "serial_sig" in steps:
        print("Running in serial!")
        significance_find.serial_significance_find(param_file)
    if "label" in steps:
        find_labels.find_labels(params)
    return


def multiple_main(param_file, params):
    print("Multiple Run!")
    multi_stage = params["multiple"]["stage"]
    print(f"multiple from stage: {multi_stage}")
    multi_list = paraf.list_files(params["data"][multi_stage])
    n_multi_current = params["multiple"].get("current", 0)
    N_samples = len(multi_list)
    for n_multi in range(n_multi_current, N_samples):
        multi_param_file = paraf.multi_create_param_file(
            param_file, n_multi=n_multi)
        main(multi_param_file)
        os.remove(multi_param_file)
    return


if __name__ == "__main__":
    param_file = sys.argv[1]
    main(param_file)
