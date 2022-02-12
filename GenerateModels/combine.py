#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:27:33 2020

@author: shruti
"""

import pandas as pd
import os, glob


#all_files = ["1314/ML/Awards1314_ml.csv", "1314/CV/Awards1314_cv.csv", "1314/NLP/Awards1314_nlp.csv"]

#all_files = glob.glob(os.path.join("1314/CV/Awards1314_cv_*.csv"))
all_files = glob.glob(os.path.join("SBIR/1415/SBIR_Awards1415_*.csv"))
for f in all_files:
    print(f)
    print(len(pd.read_csv(f, sep=',', encoding ='utf-8')))
df_merged = (pd.read_csv(f, sep=',', encoding ='utf-8') for f in all_files)
df_merged   = pd.concat(df_merged, ignore_index=True)
df_merged.to_csv( "SBIR/1415/SBIR_Awards1415.csv")


