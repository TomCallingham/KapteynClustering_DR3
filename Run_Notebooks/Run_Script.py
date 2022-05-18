from KapteynClustering.Main import main
import sys
try:
    param_file = sys.argv[1]
    main(param_file)
except Exception:
    main("./gaia_params.yaml")
