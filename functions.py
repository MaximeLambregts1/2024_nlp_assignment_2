import re
from nltk.stem import WordNetLemmatizer, PorterStemmer

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

def lemmatize_list(text):
        lemmatizer = WordNetLemmatizer()
        lemmatized_text = [lemmatizer.lemmatize(word) for word in text]
        return lemmatized_text
    
def stem_list(text):
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(word) for word in text]
    return stemmed_text