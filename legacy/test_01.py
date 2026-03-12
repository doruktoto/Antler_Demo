from bsm2_python import BSM2OL

# initialize the BSM2 Open Loop model
bsm2_ol = BSM2OL(data_out = "output_data_300d.csv",evaltime = 300)

# run the simulation
bsm2_ol.simulate()