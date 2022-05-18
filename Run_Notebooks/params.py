def get_AU_data_params( lv =4, halo_n=5):
    base= f"Lv{lv}Au{halo_n}"
    print(base)
    outputfolder = '/cosma5/data/dp004/dc-call1/Auriga_Analysis/' + \
            f'level{lv}_MHD/halo_{halo_n}/snap_127/'
    # snaptext = '{:03d}'.format((snap)) % (Type, Halo_n, snaptext)
    # Try to load expansions
    pot_sphere = outputfolder + 'axi_sphere.coef_mul'
    pot_disc = outputfolder + 'axi_disc.coef_cylsp'

    AU_data_params= {
    "data_folder" : base+"/",
    "pot_name": [ pot_sphere, pot_disc ],
    "result_path" : "/cosma/home/dp004/dc-call1/scripts/Python/AuClustering/KapteynClustering_DR3/Data/",
    "original_data": base + "_Stars",
    "base_data": base+"_toomre180",
    "base_dyn": base+"_Dyn", #Dynamics and scaled
    "sample": base+"_Sample", #Toomre Selection and others,
    "art": base+"_Art",
    "cluster": base+ "_Cluster",
    "sig": base + "_Significance"
    }
    return AU_data_params

def get_test_gaia_data_params(pot_name="H99"):
    "Could also be McMillan17.ini"
    base= "test_gaia"
    test_gaia_data_params= {
    "result_path" : "/Users/users/callingham/Projects/Clustering/KapteynClustering_DR3/Data/",
    "data_folder" : "test_gaia/",
    "base_data": base+"_Stars",
    "base_dyn": base+"_Dyn", #Dynamics and scaled
    "sample": base+"_Sample", #Toomre Selection and others,
    "art": base+"_Art",
    "cluster": base+ "_Cluster",
    "sig": base + "_Significance",
    "pot_name":pot_name
    }
    return test_gaia_data_params

import os
home = os.path.expanduser("~")

gaia2 = ("/Users/users" in home)
auriga = ("cosma" in home)
cosma = ("cosma" in home)

if gaia2 is True:
    print("Running on gaia2!")
    data_params = get_test_gaia_data_params()
else:
    print("Running on cosma, using Auriga")
    data_params = get_AU_data_params()
