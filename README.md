# NLP-scraper

## Project Goal
The goal of this project is to build an NLP-enriched platform for news analysis. The platform will detect relevant information, entities, topics, and sentiments from news articles, while also identifying potential environmental scandals.

## Project Structure
- `data/`: Contains datasets and scraped news articles.
- `nlp_enriched_news.py`: Main script for processing news articles.
- `requirements.txt`: List of required packages.
- `results/`: Directory for saving output files, including:
  - `enhanced_news.csv`: Processed results of news articles.
  - `learning_curves.png`: Plot of model training performance.
  - `topic_classifier.pkl`: Trained model for topic detection.
- `scraper_news.py`: Script for scraping news articles from a source.

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/Daniila-vv/nlp-scraper.git
   ```


2.	Set up a virtual environment and install dependencies:

    ```python -m venv .venv
    source .venv/bin/activate  # For Windows use .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.	Run the scraper to gather news articles:

    ```python scraper_news.py
    ```

4.	Process the collected articles:

    ```python nlp_enriched_news.py
    ```

## Analysis Overview

### The NLP engine processes news articles by:

1.	Entity Detection: Identifying organizations within the text using SpaCy.
2.	Topic Detection: Classifying articles into categories such as Tech, Sport, Business, Entertainment, or Politics using a trained classifier.
3.	Sentiment Analysis: Evaluating the sentiment of articles (positive, negative, or neutral) using NLTK’s sentiment analysis tools.
4.	Scandal Detection: Detecting potential environmental disasters by calculating distances between keyword embeddings and article embeddings.

### Distance Calculation

For scandal detection, we compute the distance between the embeddings of specific keywords related to environmental disasters (e.g., pollution, deforestation) and the embeddings of sentences containing identified entities. This approach leverages Word2Vec for generating meaningful word embeddings, allowing for a more nuanced understanding of how closely articles relate to environmental concerns.

The distance metric chosen for this project is the Euclidean distance, which quantifies how far apart the keyword embeddings are from the article embeddings. A smaller distance indicates a closer semantic relationship, suggesting that the article may discuss relevant environmental issues.

### Metrics

The project calculates and saves a unified metric for all distances per article in the output file. This metric enables easy identification of the top 10 articles that are most relevant to environmental scandals based on their calculated distances.

### Conclusion

The main objectives of the project have been achieved, including entity detection, topic classification, sentiment analysis, and scandal detection. However, further refinement is needed to enhance the accuracy and reliability of the results.