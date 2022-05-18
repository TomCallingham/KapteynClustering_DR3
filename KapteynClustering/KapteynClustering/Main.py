import sys
try:
    from .RunSteps import sample_prepare, art_data, single_linkage, significance_find, find_labels
    import data_funcs as dataf
except Exception:
    from KapteynClustering.RunSteps import sample_prepare, art_data, single_linkage, significance_find, find_labels
    import KapteynClustering.data_funcs as dataf

def main(param_file):
    print("Running KC script, using params found:")
    print(param_file, "\n")
    params = dataf.read_param_file(param_file)
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

if __name__ == "__main__":
    param_file = sys.argv[1]
    main(param_file)
