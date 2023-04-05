# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:16:27 2023

@author: bksat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv('API_19_DS2_en_csv_v2_4902199.csv', skiprows=4)

file.head(5)