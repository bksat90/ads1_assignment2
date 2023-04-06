# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:27:23 2023

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
    indicator = ['SP.URB.TOTL', 'SP.POP.TOTL', 'SH.DYN.MORT', 'ER.H2O.FWTL.K3',
                 'AG.LND.FRST.K2', 'AG.LND.ARBL.ZS', 'AG.LND.AGRI.K2',
                 'EN.ATM.CO2E.KT', 'EG.USE.ELEC.KH.PC']
    df = df.loc[df['Indicator Code'].isin(indicator)]
    return df


def Preprocess(df):
    """
    This function preprocesses the data
    """
    df.drop('Country Code', axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.T
    # df = df.iloc[: , :-1]
    return df


def ElectricityConsumption(df):
    """
    """
    econs_df =  df.loc[df['Indicator Code'] == 'EG.USE.ELEC.KH.PC']
    econs_df.drop(['Indicator Name', 'Indicator Code'],
                  axis=1, inplace=True)
    econs_df.reset_index(drop=True, inplace=True)
    econs_df = econs_df.iloc[: , 1:]
    econs_df = econs_df.T
    econs_df = econs_df.rename(columns=econs_df.iloc[0])
    econs_df.drop(labels=['Country Name'],axis=0, inplace=True)
    econs_df.reset_index(inplace=True)
    return econs_df


def HeatmapPreprocess(df, year):
    """
    """
    indicator = ['SP.URB.TOTL', 'SP.POP.TOTL', 'SH.DYN.MORT', 'ER.H2O.FWTL.K3',
           'AG.LND.FRST.K2', 'AG.LND.ARBL.ZS', 'AG.LND.AGRI.K2',
           'EN.ATM.CO2E.KT']
    hdf =  df.loc[df['Indicator Code'].isin(indicator)]
    hdf.drop(['Indicator Name', 'Indicator Code'],
                  axis=1, inplace=True)
    # hdf.reset_index(drop=True, inplace=True)
    # # econs_df = econs_df.T
    # econs_df = econs_df.rename(columns=econs_df.iloc[0])
    # econs_df.drop(labels=['Country Name'],axis=0, inplace=True)
    
    return


    

# Reads data from the world bank climate data
df = pd.read_csv('API_19_DS2_en_csv_v2_4902199.csv', skiprows=4)
# filters the data for the required countries
df = FilterData(df)
df = Preprocess(df)
df.to_csv('file.csv')
# edf = ElectricityConsumption(df)
# # # hdf = 



# edf['index'] = edf['index'].astype('int')
# #edf['index']= pd.to_datetime(edf['index'], format='%Y')
# edf = edf[edf['index'] > 1990]
# edf = edf[edf['index'] < 2015]
# edf['index']= pd.to_datetime(edf['index'], format='%Y')

# # line plot
# plt.figure(figsize=(8,8), dpi=400)
# plt.plot(edf['index'], edf['China'], label='China')
# plt.plot(edf['index'], edf['Germany'], label='Germany')
# plt.plot(edf['index'], edf['France'], label='France')
# plt.plot(edf['index'], edf['UK'], label='UK')
# plt.plot(edf['index'], edf['India'], label='India')
# plt.plot(edf['index'], edf['Russia'], label='Russia')
# plt.plot(edf['index'], edf['US'], label='US')
# plt.legend()
# plt.xlabel('Years')
# plt.ylabel('kilo Watt hour per capita')
# plt.title('Electric power consumption between 1990 and 2014new')
# plt.show()


# # # 