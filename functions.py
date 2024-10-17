import re

def filter_articles(df, keywords, col='content', case=False):
    """
    Filter articles by keywords
    The function will return articles that contain at least one of the keywords
    """
    df = df[df[col].str.contains('|'.join(keywords), case=case)]
    return df

def clean_text(text):
    text = text.lower() # lowercase text
    text = re.sub(r'\\n', ' ', text) # remove new line characters
    text = re.sub(r'\d+', '', text) # remove digits
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation
    text = re.sub(r'\s+', ' ', text) # remove extra spaces
    return text