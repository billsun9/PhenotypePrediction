import requests
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd

def scrape(url):
    try:
        res = requests.get(url)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # STILL WIP
        table = soup.find('tbody', {"id": "all_results"})
# =============================================================================
#         closing_tags = table.find_all(["p","table","h2","h3"])[::-1][:3]
#         for closing_tag in closing_tags:
#             closing_tag.decompose()
#         toc = table.find("div", {"id": "toc"})
#         if toc:
#             toc.decompose()
# =============================================================================
        return table
        
    except requests.exceptions.HTTPError as http_err:
        raise SystemExit(http_err)
        print('HTTP error:', str(http_err))
    except Exception as err:
        print('Some other error:', str(err))
        

url = 'http://synergistic.aging-research.group/roundworm/'

page = scrape(url)
# %%
print(page.find('tr').prettify())