import pandas as pd
import os
import shutil
from pathlib import Path
import re

# Read the Excel file
df = pd.read_csv("PollutionData/poldata.csv", skiprows=2)

col_name = "RANDWICK OZONE 24h average [pphm]"

# Find the start index of the target column
start_idx = df.columns.get_loc(col_name)

# Keep the first column ("Date") + from target column onwards
df_subset = pd.concat([df.iloc[:, [0]], df.iloc[:, start_idx:]], axis=1)

df

