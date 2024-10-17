#import modules
import pandas as pd
import gensim
from ast import literal_eval
from FuzzyTM import FLSA_W
from nltk.tokenize import word_tokenize
import re
# from unidecode import unidecode
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from collections import Counter

#import data
# import data
FILE_PATH = 'data/nvidia_articles.csv'
data = pd.read_csv(FILE_PATH,
                   usecols=['content'],
                   )['content'].tolist()

flsaW = FLSA_W(input_file = data,
               num_topics=10,
               num_words=10,
               )

flsaW.get_vocabulary_size()
pwgt, ptgd = flsaW.get_matrices() # THIS TRAINS THE MODEL
print(flsaW.show_topics())
for topic in flsaW.show_topics(representation='words'):
    print(topic)
print(flsaW.get_coherence_score())
print(flsaW.get_diversity_score())
print(flsaW.get_interpretability_score())
