# import modules
import pandas as pd
from nltk.corpus import stopwords
from functions import *

if __name__ == '__main__':
    # import data
    df = pd.read_csv('./data/us_equities_news_dataset.csv')

    # drop url and article_id columns
    df = df.drop(columns=['url', 'article_id'])

    # filter articles with missing content
    df = df.dropna(subset=['content'])

    # filter Nvidia articles
    df = filter_articles(df, ['Nvidia', 'nvda'])

    # save original content
    df['original_content'] = df['content']

    # clean text
    df['content'] = df['content'].apply(lambda x: clean_text(x))

    # tokenize text
    df['content'] = df['content'].apply(lambda x: x.split())

    # remove stop words
    stop_words = set(stopwords.words('english'))
    df['content'] = df['content'].apply(lambda x: [word for word in x if word not in stop_words])
    
    # stem content
    df['stemmed_content'] = df['content'].apply(lemmatize_list)

    # lemmatize content
    df['lemmatized_content'] = df['content'].apply(stem_list)

    # save file
    df.to_csv('./data/nvidia_articles.csv', index=False)