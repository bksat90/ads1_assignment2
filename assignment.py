# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:16:27 2023

@author: bksat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def FilterData(df):
    """
    This function filters the dataframe for the required countries such as US,
    UK, France, India, China, Germany, Russia and keeps only the required
    fields.
    """
    # filter out data for US, UK, France, India, China, Germany, Russia
    countries = ['United States', 'United Kingdom', 'France', 'India', 'China',
             'Germany', 'Russian Federation', ]
    df = df.loc[df['Country Name'].isin(countries)]
    
    # required indicators
    indicator = ['SP.URB.TOTL', 'SP.POP.TOTL', 'SH.DYN.MORT', 'SH.DYN.MORT', 
                 'ER.H2O.FWTL.K3', 'EN.ATM.GHGT.KT.CE', 'EN.ATM.CO2E.KT',
                 'EN.ATM.CO2E.SF.KT', 'EN.ATM.CO2E.LF.KT', 'EN.ATM.CO2E.GF.KT',
                 'EG.USE.ELEC.KH.PC', 'EG.ELC.RNEW.ZS', 'AG.LND.FRST.K2',
                 'AG.LND.ARBL.ZS', 'AG.LND.AGRI.K2']
    df = df.loc[df['Indicator Code'].isin(indicator)]
    return df

def Preprocess(df):
    """
    This function preprocesses the data
    """
    df.drop('Country Code', axis=1, inplace=True)
    df.fillna(0, inplace=True)
    return df

# Reads data from the world bank climate data
df = pd.read_csv('API_19_DS2_en_csv_v2_4902199.csv', skiprows=4)
# filters the data for the required countries
df = FilterData(df)
df = Preprocess(df)