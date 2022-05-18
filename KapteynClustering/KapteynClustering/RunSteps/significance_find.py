import os
import KapteynClustering as KC
def significance_find(param_file):
    module_folder = os.path.dirname(KC.__file__)
    script_file = module_folder + "/Run_Significance.py"
    os.system(f"python {script_file} {param_file}")
    print("Finished")
    return
def serial_significance_find(param_file):
    module_folder = os.path.dirname(KC.__file__)
    script_file = module_folder + "/serial_Run_Significance.py"
    os.system(f"python {script_file} {param_file}")
    print("Finished")
    return
