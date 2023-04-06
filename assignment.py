# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:16:27 2023

@author: bksat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import cluster_tools as ct

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


def ElectricityConsumption(df):
    """
    """
    econs_df =  df.loc[df['Indicator Code'] == 'EG.USE.ELEC.KH.PC']
    econs_df.drop(['Indicator Name', 'Indicator Code'],
                  axis=1, inplace=True)
    econs_df.reset_index(drop=True, inplace=True)
    econs_df = econs_df.T
    econs_df = econs_df.rename(columns=econs_df.iloc[0])
    econs_df.drop(labels=['Country Name'],axis=0, inplace=True)
    econs_df.rename(columns={'United Kingdom': 'UK',
                             'Russian Federation': 'Russia',
                             'United States': 'US'}, inplace=True)
    econs_df.reset_index(inplace=True)
    econs_df.drop(econs_df.tail(1).index,inplace=True)
    return econs_df


def HeatmapPreprocess(df, country):
    """
    """
    hdf = df.loc[df['Country Name'] == country]
    indicator = ['SP.URB.TOTL', 'SP.POP.TOTL', 'SH.DYN.MORT', 'ER.H2O.FWTL.K3',
           'AG.LND.FRST.K2', 'AG.LND.ARBL.ZS', 'AG.LND.AGRI.K2',
           'EN.ATM.CO2E.KT']
    hdf =  hdf.loc[hdf['Indicator Code'].isin(indicator)]
    hdf.drop(['Country Name', 'Indicator Code'], axis=1, inplace=True)
    hdf.reset_index(drop=True, inplace=True)
    #hdf = hdf[['Indicator Name', str(year)]]
    #hdf = hdf.set_index(['Indicator Name'])
    hdf = hdf.T
    hdf.reset_index(inplace=True)
    hdf = hdf.rename(columns=hdf.iloc[0])
    #hdf.drop(labels=['Indicator Name'],axis=0, inplace=True)
    hdf.drop(0, inplace=True)
    hdf.drop(hdf.tail(3).index,inplace=True)
    hdf.rename(columns=
                {'Indicator Name': 'Year',
                'Population, total': 'Total Population',
                'Mortality rate, under-5 (per 1,000 live births)':
                    'Mortality rate',
                'Annual freshwater withdrawals, total (billion cubic meters)'
                    : 'Annual freshwater withdrawals',
                'CO2 emissions (kt)': 'CO2 emissions',
                'Forest area (sq. km)': 'Forest area',
                'Arable land (% of land area)': 'Arable land',
                'Agricultural land (sq. km)': 'Agricultural land'},
                inplace=True)
    
    hdf['Year'] = hdf['Year'].astype('int')
    hdf = hdf[hdf['Year'] >= 1990]
    hdf.reset_index(drop=True, inplace=True)
    hdf['Year'] = hdf['Year'].astype('object')
    
    columns = list(hdf.columns)
    for col in columns:
        if col != 'Year':
            hdf[col] = hdf[col].astype('float64')
    
    return hdf


#main code
# Reads data from the world bank climate data
df = pd.read_csv('API_19_DS2_en_csv_v2_4902199.csv', skiprows=4)
# filters the data for the required countries
df = FilterData(df)
df = Preprocess(df)

#electricity
# edf = ElectricityConsumption(df)

# edf['index'] = edf['index'].astype('int')
# edf = edf[edf['index'] > 1990]
# edf = edf[edf['index'] < 2015]
# edf['index']= pd.to_datetime(edf['index'], format='%Y')

# # line plot
# plt.figure(figsize=(8,8), dpi=400)
# plt.plot(edf['index'], edf['China'], label='China')
# plt.plot(edf['index'], edf['Gesrmany'], label='Germany')
# plt.plot(edf['index'], edf['France'], label='France')
# plt.plot(edf['index'], edf['UK'], label='UK')
# plt.plot(edf['index'], edf['India'], label='India')
# plt.plot(edf['index'], edf['Russia'], label='Russia')
# plt.plot(edf['index'], edf['US'], label='US')
# plt.legend()
# plt.xlabel('Years')
# plt.ylabel('kilo Watt hour per capita')
# plt.title('Electric power consumption between 1990 and 2014')
# plt.show()

# correlation map
hdf_Ge = HeatmapPreprocess(df, 'Germany')
hdf_China = HeatmapPreprocess(df, 'China')
ct.map_corr(hdf_Ge)
ct.map_corr(hdf_China)


#hdf.to_csv('file.csv')

import seaborn as sb
dataplot = sb.heatmap(hdf_China.corr(), cmap="YlGnBu")