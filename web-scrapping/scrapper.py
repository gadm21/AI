
'''
HOW TO RUN: python scrapper.py --num_news <number of items to scrape>

EXAMPLE: python scrapper.py --num_news 100
'''




import math
import json 
import argparse 
import unicodedata

import bs4 
from urllib.request import urlopen as uReq 
from bs4 import BeautifulSoup as soup 
import requests 



url = 'https://news.un.org/en/news/region/middle-east?page=' 
json_path = 'results.json'

def get_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--num_news', default=10)
    args = parser.parse_args() 
    return args 

def get_content(n):

    def get_page_content(url):
        #client = uReq(url) 
        #html_page = client.read()
        
        html_page = requests.get(url).content.decode('utf8')
        html_page = html_page.encode('ascii', 'ignore')
        soup_page = soup(html_page.decode('utf-8','ignore'), 'html.parser') 
        page_titles = soup_page.findAll('h1', {'class':'story-title'})
        page_summaries = soup_page.findAll('div', {'class':'news-body'})
        page_urls = list(map(lambda title: title.a['href'], page_titles))
        page_titles = list(map(lambda title: title.a.text, page_titles))
        page_summaries = list(map(lambda s: s.p.text, page_summaries))
        page_summaries = [unicodedata.normalize("NFKD", summary) for summary in page_summaries]

        #client.close()  
        return page_urls[:n], page_titles[:n], page_summaries[:n]

    n_pages = int(math.ceil(n/10)) 
    titles = [] 
    urls = [] 
    summaries = []
    for page in range(n_pages) :
        page_url = url+str(page) 
        page_urls, page_titles, page_summaries = get_page_content(page_url) 
        titles += page_titles 
        urls += page_urls 
        summaries += page_summaries
    return urls, titles, summaries

def get_complete_urls(urls):
    website = 'https://news.un.org' 
    complete_urls = [website+url for url in urls]
    return complete_urls 

def save_to_json(json_path, titles, urls, summaries):

    items = []
    for i in range(len(titles)):
        news_item = {'title': titles[i], 'url':urls[i], 'summary':summaries[i]}
        items.append(news_item)
    
    
    with open(json_path, 'w') as f :
        json.dump(items, f, indent=1) 

def main():
    args = get_args() 

    urls, titles, summaries = get_content(int(args.num_news)) 
    urls = get_complete_urls(urls) 
    
    save_to_json(json_path, titles, urls, summaries)



if __name__ == '__main__':
    main()
