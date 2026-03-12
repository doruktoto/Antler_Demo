from bsm2_python import BSM2OLEM

bsm2_olem = BSM2OLEM()

bsm2_olem.simulate()

# get the cumulated cash flow of the plant
cash_flow = bsm2_olem.economics.cum_cash_flow