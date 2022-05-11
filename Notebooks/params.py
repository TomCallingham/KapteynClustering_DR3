base= "Lv4Au5"
AU5_data_params= {
"result_path" : "/cosma/home/dp004/dc-call1/scripts/Python/AuClustering/KapteynClustering_DR3/Data/",
"data_folder" : "Au5Lv4/",
"base_data": base+"_Stars",
"base_dyn": base+"_Dyn", #Dynamics and scaled
"sample": base+"_Sample", #Toomre Selection and others,
"art": base+"_Art",
"cluster": base+ "_Cluster",
"sig": base + "_Significance"
}

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
}
test_gaia_data_params["pot_file"] = test_gaia_data_params["result_path"] +"McMillan17.ini"

import os
home = os.path.expanduser("~")

gaia2 = ("/Users/users" in home)
auriga = ("cosma" in home)
cosma = ("cosma" in home)

if gaia2 is True:
    print("Running on gaia2!")
    data_params = test_gaia_data_params
else:
    print("Running on cosma, using AU5")
    data_params = AU5_data_params

