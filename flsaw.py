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

# import data, source: https://towardsdatascience.com/fuzzytm-a-python-package-for-fuzzy-topic-models-fd3c3f0ae060
df = pd.read_csv('data/nvidia_articles.csv', converters={'content': literal_eval})

#Convert data to list of lists
data = df['content'].values.tolist()

#Create word counter
all_words = [word for doc in data for word in doc]
word_freq = Counter(all_words)


#First iteration of making sense
nr_of_topics = list(range(1,11))
dict_coherence_score = {}
topic_dict = {}

for i in nr_of_topics:
    flsaW = FLSA_W(input_file=data,
                   num_topics=i,
                   num_words=10,
                   )
    flsaW.get_vocabulary_size()
    pwgt, ptgd = flsaW.get_matrices()
    topic_dict[i] = {}

    topics = flsaW.show_topics(representation='words')  # haal de topics op in woordrepresentatie

    # Loop door de topics en sla ze op
    for topic_num, topic in enumerate(topics):
        topic_dict[i][topic_num] = topic  # sla elk topic en bijbehorende woorden op

    dict_coherence_score[i] = flsaW.get_coherence_score()

print(dict_coherence_score)
print(topic_dict)
print(topic_dict[10])


#First iteration of word removal
words_to_remove = ['also', 'p', 'whileolympusshares', 'upnoteus', 'foo', 'chee']
# Check frequency of words
word_counts = {word: word_freq[word] for word in words_to_remove}

print("Word frequencies of specific words:", word_counts)

data_iteration_2 = [[word for word in document if word not in words_to_remove] for document in data]