"""
Created on May 3 2021
@author: Jay Ram
"""

import requests
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import csv

def csvprint(table):
    table.to_csv('file.csv')

def scrape(url):
    try:
      
        table = pd.DataFrame()
        
        for i in range(36):
            purl = url + "?page=" + str(i)
            print(purl)
            r = requests.get(purl)
            df_list = pd.read_html(r.text) # this parses all the tables in webpages to a list
            df = df_list[0]
            
            print(type(df))
            print(df.shape)
          
            if(i == 0):
                table = df
            else:
                table = pd.concat([table,df[1:]],ignore_index=True)
            print(table.shape)
            
        return table
        
    except requests.exceptions.HTTPError as http_err:
        raise SystemExit(http_err)
        print('HTTP error:', str(http_err))
    except Exception as err:
        print('Some other error:', str(err))
        

url = 'http://synergistic.aging-research.group/roundworm/'

table = scrape(url)
# %%
print(table)
csvprint(table)
