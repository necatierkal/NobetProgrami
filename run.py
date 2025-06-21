# import required packages
import os
import mip_setup as mip_setup
import mip_solve as mip_solve
import openpyxl
import pandas as pd

# read user inputs
inputs_directory = os.path.join('inputs', 'inputs_to_load.xlsx')  # os.getcwd(),
inputs_dict = mip_setup.read_inputs(inputs_directory)

# prepare
mip_inputs = mip_setup.InputsSetup(inputs_dict)
mip_solve.mathematical_model_solve(mip_inputs)

