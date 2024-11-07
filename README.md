# 2024_nlp_assignment_2

This github contains the code for three different topic modeling approaches to retrieve topics from the us_equities_news_dataset related to NVIDIA.
The code is a mix of jupyter notebooks (.ipynb) and python scripts (.py). In this repository you will find the following files

      preprocessing.py
        This script contains the function to filter the NVIDIA articles and preprocess the text in the articles. It returns a CSV file with the original data and two extra columns, the lemmatized text and the stemmed text.

      functions.py
        This script contains the different functions that are used in the preprocessing and in the jupyter notebooks that follow. These contain basic functions like computing coherence scores, interpretability scores and plotting certain results

      LDA.ipynb
        This jupyter notebook contains the code for creating the topic model LDA. It includes importing the data, training the model and evaluating the results

      BERT.ipynb
        This jupyter notebook contains the code for creating the topic model BERtopic. It includes importing the data, training the model and evaluating the results

      flsaw.py
        This jupyter notebook contains the code for creating the topic model FLSA-W. It includes importing the data, training the model and evaluating the results


