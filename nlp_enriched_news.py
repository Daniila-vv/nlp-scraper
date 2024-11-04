import pandas as pd
import spacy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors  # Импортируем KeyedVectors для работы с Word2Vec

# Загружаем модель SpaCy для обнаружения сущностей
nlp = spacy.load("en_core_web_sm")

# Загружаем предобученный анализатор настроений
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Функция для загрузки модели Word2Vec
def load_word2vec_model():
    model = KeyedVectors.load_word2vec_format('results/GoogleNews-vectors-negative300.bin', binary=True)  # Укажите путь к модели
    return model

# Загружаем размеченные данные для классификации темы
def load_topic_data():
    train_data = pd.read_csv('data/bbc_news_train.txt', sep=',', header=0, names=['ArticleId', 'Text', 'Category'])
    test_data = pd.read_csv('data/bbc_news_tests.txt', sep=',', header=0, names=['ArticleId', 'Text', 'Category'])
    return train_data, test_data

# Предобработка текста для классификации темы
def preprocess_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

# Обучение классификатора тем
def train_topic_classifier(train_data):
    vectorizer = CountVectorizer(preprocessor=preprocess_text)
    X_train = vectorizer.fit_transform(train_data['Text'])
    y_train = train_data['Category']

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    joblib.dump(clf, 'results/topic_classifier.pkl')
    joblib.dump(vectorizer, 'results/vectorizer.pkl')

    # Оценка модели и создание графика
    train_accuracy = []
    test_accuracy = []

    for i in range(1, 11):
        X_train_sub, X_test, y_train_sub, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        clf.fit(X_train_sub, y_train_sub)
        
        train_accuracy.append(clf.score(X_train_sub, y_train_sub))
        test_accuracy.append(clf.score(X_test, y_test))

    # Строим график
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), train_accuracy, label='Train Accuracy', marker='o')
    plt.plot(range(1, 11), test_accuracy, label='Test Accuracy', marker='o')
    plt.title('Learning Curves')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('results/learning_curves.png')

# Обнаружение сущностей в статье
def detect_entities(article):
    doc = nlp(article)
    companies = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return companies

# Определение темы статьи
def detect_topic(article, vectorizer, clf):
    X = vectorizer.transform([article])
    topic = clf.predict(X)[0]
    return topic

# Проведение анализа настроений
def analyze_sentiment(article):
    score = sia.polarity_scores(article)
    sentiment = score['compound']
    return sentiment

# Вычисление расстояния скандала на основе Word2Vec
def calculate_scandal_distance(article, keywords, model):
    distances = []
    article_vector = np.mean([model[word] for word in article.lower().split() if word in model.key_to_index], axis=0)
    
    for keyword in keywords:
        if keyword in model.key_to_index:
            word_vector = model[keyword]
            distance = np.linalg.norm(word_vector - article_vector)
            distances.append(distance)
    
    return np.min(distances) if distances else float('inf')

def process_articles(articles, model):
    results = []

    # Загружаем классификатор тем
    clf = joblib.load('results/topic_classifier.pkl')
    vectorizer = joblib.load('results/vectorizer.pkl')

    # Определяем ключевые слова, связанные со скандалом
    scandal_keywords = [
        'pollution', 'air pollution', 'water pollution', 'soil pollution', 'plastic pollution',
        'industrial waste', 'hazardous waste', 'chemical spills', 'toxic waste', 'sewage contamination',
        'climate change', 'global warming', 'greenhouse gases', 'carbon footprint', 'fossil fuels',
        'climate crisis', 'climate emergency', 'extreme weather', 'rising sea levels', 'heatwaves',
        'drought', 'flooding', 'deforestation', 'habitat destruction', 'biodiversity loss', 
        'endangered species', 'extinction', 'habitat fragmentation', 'overfishing', 'illegal logging', 
        'wildlife trafficking', 'ecosystem degradation', 'resource depletion', 'overconsumption', 
        'unsustainable practices', 'land degradation', 'desertification', 'water scarcity', 
        'oil spills', 'mining impact', 'fracking', 'natural disasters', 'earthquakes', 'tsunamis', 
        'wildfires', 'volcanic eruptions', 'hurricanes', 'tornadoes', 'melting ice caps', 
        'ocean acidification', 'coral bleaching', 'invasive species', 'environmental justice', 
        'climate refugees', 'environmental regulations', 'conservation efforts', 'sustainability', 
        'green policies', 'environmental activism', 'eco-terrorism', 'corporate responsibility', 
        'ecological footprint', 'ecological imbalance', 'toxic substances', 'health risks', 
        'pollution control', 'environmental impact', 'renewable resources', 'sustainable development', 
        'carbon emissions', 'soil erosion', 'water contamination'
    ]

    for index, row in tqdm(articles.iterrows(), total=articles.shape[0]):
        url = row['url']
        date = row['date']
        headline = row['headline'] if pd.notna(row['headline']) else ""  # Обрабатываем NaN
        body = row['body'] if pd.notna(row['body']) else ""  # Обрабатываем NaN

        # Обнаружение сущностей
        orgs = detect_entities(headline + " " + body)

        # Определение темы
        topic = detect_topic(headline + " " + body, vectorizer, clf)

        # Анализ настроений
        sentiment = analyze_sentiment(headline + " " + body)

        # Вычисление расстояния скандала
        scandal_distance = calculate_scandal_distance(headline + " " + body, scandal_keywords, model)

        # Сохранение результатов
        results.append({
            'unique_id': index,
            'url': url,
            'date': date,
            'headline': headline,
            'body': body,
            'orgs': orgs,
            'topic': topic,
            'sentiment': sentiment,
            'scandal_distance': scandal_distance,
            'top_10': False  # Место для логики топ 10
        })

    # Преобразуем результаты в DataFrame
    results_df = pd.DataFrame(results)

    # Здесь можно добавить логику для фильтрации и пометки топ 10 статей на основе расстояния скандала
    top_10_indices = results_df.nsmallest(10, 'scandal_distance').index
    results_df.loc[top_10_indices, 'top_10'] = True

    # Сохраняем обогащенные данные в CSV файл
    results_df.to_csv('results/enhanced_news.csv', index=False)

if __name__ == "__main__":
    # Загружаем данные для обучения
    train_data, test_data = load_topic_data()  # Загружаем данные для обучения
    train_topic_classifier(train_data)  # Обучаем классификатор на тренировочных данных

    # Загружаем предобученную модель Word2Vec
    word2vec_model = load_word2vec_model()  # Укажите путь к вашей модели Word2Vec

    # Загружаем статьи из CSV
    articles = pd.read_csv('data/guardian_articles.csv')  # Настройте путь по мере необходимости
    process_articles(articles, word2vec_model)