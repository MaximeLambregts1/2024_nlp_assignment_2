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
import matplotlib as plt

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


#Iteration 2
flsaW = FLSA_W(input_file = data_iteration_2,
               num_topics = 10,
               num_words = 10)
flsaW.get_vocabulary_size()
pwgt, ptgd = flsaW.get_matrices()
topics = flsaW.show_topics(representation='words')
print(topics)
flsaW.get_coherence_score(input_file = data_iteration_2, topics = topics)
flsaW.get_diversity_score(topics = topics)
flsaW.get_interpretability_score(input_file = data_iteration_2, topics = topics)

for topic in topics:
    print(topic)

def replace_and_remove_words_in_data(data, replace_words, replacement_words, words_to_remove):
    # Maak een woordenboek voor vervangende woorden
    replacement_dict = dict(zip(replace_words, replacement_words))

    # Verwerk de documenten
    for i, document in enumerate(data):
        # Vervang de woorden en verwijder woorden uit words_to_remove
        data[i] = [replacement_dict.get(word, word) for word in document if word not in words_to_remove]

    return data


replace_words = ['nvda', 'minds', 'books', 'corp', 'amzn', 'kem', 'sept']
replacement_words = ['nvidia', 'mind', 'book', 'corporation', 'amazon', 'kemet', 'september']

words_to_remove = ['equitiesasian']

#Create word counter
all_words = [word for doc in data_iteration_2 for word in doc]
word_freq = Counter(all_words)
# Check frequency of words
word_counts = {word: word_freq[word] for word in words_to_remove}

print("Word frequencies of specific words:", word_counts)

data_iteration_3 = replace_and_remove_words_in_data(data_iteration_2, replace_words, replacement_words, words_to_remove)

#Iteration 3
flsaW = FLSA_W(input_file = data_iteration_3,
               num_topics = 10,
               num_words = 10)
flsaW.get_vocabulary_size()
pwgt, ptgd = flsaW.get_matrices()
topics = flsaW.show_topics(representation='words')
print(topics)
flsaW.get_coherence_score(input_file = data_iteration_3, topics = topics)
flsaW.get_diversity_score(topics = topics)
flsaW.get_interpretability_score(input_file = data_iteration_3, topics = topics)

for topic in topics:
    print(topic)

#Iteration 4
nr_of_topics = list(range(1,11))
dict_coherence_score = {}
topic_dict = {}

for i in nr_of_topics:
    flsaW = FLSA_W(input_file=data_iteration_3,
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

# Data voor de grafiek
keys = list(dict_coherence_score.keys())
values = list(dict_coherence_score.values())

# Maak de grafiek
plt.figure(figsize=(10, 5))
plt.bar(keys, values, color='blue')

# Voeg labels en titel toe
plt.ylabel('Coherence score')
plt.xlabel('Number of topics')
plt.title('Coherence score per topic')

# Stel de y-as in van 0 tot 1
plt.ylim(0, 1)

# Toon de grafiek
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optioneel: voeg een raster toe
plt.show()