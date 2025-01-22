import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import uuid

def fetch_news():
    urls = [
        'https://www.theguardian.com/environment/climate-crisis',
        'https://www.theguardian.com/science',
        'https://www.theguardian.com/uk/environment',
        'https://www.theguardian.com/global-development',
        'https://www.theguardian.com/uk/technology',
        'https://www.theguardian.com/uk/business',
        'https://www.theguardian.com/world',
        'https://www.theguardian.com/uk/news',
        'https://www.theguardian.com/us-news',
        'https://www.theguardian.com/world/ukraine',
        'https://www.theguardian.com/world/americas',
        'https://www.theguardian.com/world/asia',
        'https://www.theguardian.com/australia-news',
        'https://www.theguardian.com/world/europe-news',
        'https://www.theguardian.com/world/africa',
        'https://www.theguardian.com/books',
        'https://www.theguardian.com/music',
        'https://www.theguardian.com/tv-and-radio',
        'https://www.theguardian.com/lifeandstyle/health-and-wellbeing',
        'https://www.theguardian.com/food',
        'https://www.theguardian.com/culture',
        'https://www.theguardian.com/society',
        'https://www.theguardian.com/culture/comedy',
        'https://www.theguardian.com/education',
        'https://www.theguardian.com/business/economics',
        'https://www.theguardian.com/health',
        'https://www.theguardian.com/politics',
        'https://www.theguardian.com/environment/pollution',
        'https://www.theguardian.com/environment/wildlife',
        'https://www.theguardian.com/environment/climate-crisis+commentisfree/commentisfree',
        'https://www.theguardian.com/environment/energy',
        'https://www.theguardian.com/environment/climate-change',
        'https://www.theguardian.com/environment/green-living',
        'https://www.theguardian.com/environment/forests',
        'https://www.theguardian.com/environment/sea-level-rise',
        'https://www.theguardian.com/environment/global-warming',
        'https://www.theguardian.com/environment/conservation',
        'https://www.theguardian.com/environment/climate-policy',
        'https://www.theguardian.com/science/environment',
        'https://www.theguardian.com/science/climate',
        'https://www.theguardian.com/science/nature',
        'https://www.theguardian.com/science/planet-earth',
        'https://www.theguardian.com/science/space',
        'https://www.theguardian.com/science/health',
        'https://www.theguardian.com/science/technology'
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    articles = []

    for url in urls:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                text = link.get_text(strip=True)
                if href and 'theguardian.com' in href and len(text) > 20:
                    article_text = fetch_article_text(href)
                    articles.append({
                        'id': str(uuid.uuid4()),  # Generate a unique ID
                        'url': href,
                        'date': datetime.now().date(),  # Save current date
                        'headline': text,
                        'body': article_text
                    })
            print(f"Завершена обработка {url}")
        else:
            print(f"Ошибка при загрузке {url} - статус: {response.status_code}")

    # Creating a DataFrame
    news_df = pd.DataFrame(articles)
    # Remove duplicates by unique ID
    news_df = news_df.drop_duplicates(subset=['id'])  
    news_df.to_csv('data/guardian_articles.csv', index=False)
    print(f"Количество загруженных статей: {len(news_df)}")
    return news_df

def fetch_article_text(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Ошибка при загрузке статьи {url} - статус: {response.status_code}")
        return ''
    
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return ' '.join([para.get_text() for para in paragraphs])


# Run the script
news_df = fetch_news()