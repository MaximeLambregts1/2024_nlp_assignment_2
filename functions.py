import re
import pandas as pd
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Sparse2Corpus
from sklearn.feature_extraction.text import TfidfVectorizer


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

class LDA:
    def __init__(self, docs, n_min=None, p_max=None, n_topics=10, use_tfidf=False, custom_stopwords=None):
        self.docs = docs
        self.n_min = n_min
        self.p_max = p_max
        self.filterwords = []
        self.n_topics = n_topics
        self.use_tfidf = use_tfidf
        self.model = None
        self.id2word = None
        self.docs_matrix = None # tfidf or bow
        if custom_stopwords is not None:
            self.filterwords.extend(custom_stopwords)
            self.filter_docs()

    def get_filter_words(self, n_min, p_max):
        if n_min is None:
            n_min = 0
        if p_max is None:
            p_max = 1        

        all_content = list(self.docs)
        all_words = [item for row in all_content for item in row]
        unique_words = set(all_words)
        
        docs_set = self.docs.apply(set)
        n_articles = len(self.docs)
        for word in unique_words:
            n_articles_word = docs_set.apply(lambda x: word in x).sum()
            p_articles = n_articles_word / n_articles
            if n_articles_word < n_min:
                self.filterwords.append(word)
            if p_max is not None and p_articles > p_max:
                self.filterwords.append(word)
        
    def filter_docs(self):
        filterwords_set = set(self.filterwords)
        self.docs = self.docs.apply(lambda x: [word for word in x if word not in filterwords_set]) 
    
    def get_tf_idf(self):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
        tfidf = tfidf_vectorizer.fit_transform(self.docs)
        tfidf = Sparse2Corpus(tfidf, documents_columns=False)
        return tfidf
    
    def get_bow(self):
        bow = [self.id2word.doc2bow(doc) for doc in self.docs]
        return bow
     

    def train_model(self):
        # filter words
        if self.n_min is not None or self.p_max is not None:
            self.get_filter_words(n_min=self.n_min, p_max=self.p_max)
            self.filter_docs()

        # get dictionary
        self.id2word = Dictionary(self.docs)

        # get document representations
        if self.use_tfidf:
            self.docs_matrix = self.get_tf_idf()
        else:
            self.docs_matrix = self.get_bow()
        
        # train model
        self.model = LdaMulticore(corpus=self.docs_matrix,
                                                id2word=self.id2word,
                                                num_topics=self.n_topics,
                                                workers=2)

    def coherence_score(self):
        coherence_model_lda = CoherenceModel(model=self.model, texts=self.docs, dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda
    
    def diversity_score(self, n_words=10):
        all_topic_words = []
        for i in range(self.n_topics):
            topic_words = [x[0] for x in self.model.get_topic_terms(i, topn=n_words)]
            all_topic_words.extend(topic_words)
        unique_topic_words = set(all_topic_words)
        topic_var = len(unique_topic_words) / len(all_topic_words)
        return topic_var
    
    def interpretability_score(self):
        interpretability = self.coherence_score() * self.diversity_score()
        return interpretability

    def perplexity_score(self):
        perplexity = self.model.log_perplexity(self.docs_matrix)
        return perplexity

    def print_topics(self):
        for idx, topic in self.model.print_topics(-1):
            print(f'Topic: {idx} - Words: {topic}')

    def plot_topics(self, n_words=5):
        # get ncols and nrows from n_topics
        max_cols = 5
        ncols = min(self.n_topics, max_cols)
        nrows = (self.n_topics // max_cols) + 1
        if self.n_topics % max_cols == 0:
            nrows -= 1
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
        fig.tight_layout(pad=4) # add padding
        
        ax = ax.flatten()
        for topic_idx, topic in enumerate(self.model.show_topics(num_words=n_words)):
            if topic_idx >= ncols * nrows:
                break  # Only plot as many topics as subplots available
            
            topics = topic[1].split(' + ')
            topics = [topic.split('*') for topic in topics]
            topics = [[float(weight), word.strip('"')] for weight, word in topics]
            topics_df = pd.DataFrame(topics, columns=['weight', 'word'])
            
            # Plot in the appropriate subplot axis
            topics_df.plot(kind='bar', x='word', y='weight', title=f'Topic {topic_idx+1}', legend=False, ax=ax[topic_idx])
            ax[topic_idx].set_xlabel('') # remove x-axis label
            ax[topic_idx].set_xticklabels(topics_df['word'], rotation=20, ha='right') # rotate x ticks

        plt.show()