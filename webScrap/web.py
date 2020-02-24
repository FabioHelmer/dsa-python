# -*- encoding: utf-8 -*-

import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import json

url = "https://stats.nba.com/players/traditional/?DateFrom=02%2F16%2F2020&DateTo=02%2F16%2F2020&PerMode=Totals&sort=AGE&dir=1&Season=2019-20&SeasonType=All%20Star"

options = Options()
options.headless = True
driver = webdriver.Chrome(chrome_options=options)
options.quit()