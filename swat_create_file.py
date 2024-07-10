import pandas as pd
import numpy as np

import os
CWD = os.getcwd() # execute from '...\voraus-ad-dataset'

normal_data = pd.read_excel(CWD + "\\Pepper & SWAT\\SWAT\\Physical-20240613T200059Z-001\\Physical\\SWaT_Dataset_Normal_v0.xlsx", header=1)

#save normal data as csv for faster reading
normal_data.to_csv(CWD + "\\Pepper & SWAT\\SWAT\\SWaT_Dataset_Normal_v0_final.csv", index= False)