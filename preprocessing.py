import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json

def filter_and_save_nvda_articles(input_csv, output_csv):
    """"
    Filter news articles that mention 'NVIDIA' or 'NVDA' in the 'title' or 'content' columns,
    and save the filtered articles to a new CSV file.

    Parameters:
    input_csv (str): Path to the input CSV file containing the news articles.
    output_csv (str): Path to save the filtered articles as a new CSV file.

    Returns:
        -None: It saves the dataframe on the computer as a csv file
    """

    kolommen = ['id', 'ticker', 'title', 'content', 'release_date', 'provider', 'article_id']

    # read csv
    df = pd.read_csv(input_csv, usecols=kolommen)

    # Filter rows with 'NVIDIA' of 'NVDA' in 'title' of 'content' -> regular expression
    df_nvda_filtered = df[df['title'].str.contains('NVIDIA|NVDA', case=False, na=False) |
                          df['content'].str.contains('NVIDIA|NVDA', case=False, na=False) |
                          df['ticker'].str.contains('NVIDIA|NVDA', case=False, na=False)]

    # check for datetime
    if not pd.api.types.is_datetime64_any_dtype(df['release_date']):
        df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d')
        print("De 'release_date' kolom is geconverteerd naar datetime formaat.")
    else:
        print("De 'release_date' kolom is al in datetime formaat.")

    # save to csv
    df_nvda_filtered.to_csv(output_csv, index=False)

    print(f"Gefilterde artikelen opgeslagen in {output_csv}")


output_csv = 'nvda_filtered_articles.csv'
filter_and_save_nvda_articles('us_equities_news_dataset.csv', output_csv)

# preprocessing of content column (if you want to add the title to the content, do it above in the function and merge it
# such that they are both in the content column


def preprocess_content(df):
    # Initialization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Definition to split up words like pointsUS and MarketsS. It adds a space where uncapitalized and capitalized letters
    #form a word where they shouldn't
    # Verbeterde definitie om woorden als 'MarketsS' of 'pointsUS' goed te splitsen
    def split_on_uppercase(text):
        tokens = re.findall(r'[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z][a-z]+|\d+', text)
        return tokens

    def clean_text(text):
        # 1. remove \n from text
        text = re.sub(r'\\n', ' ', text)

        # 2. lowercasing
        text = text.lower()

        # 3. split in tokens on uppercase and spaces
        tokens = split_on_uppercase(text)

        # 4. remove stopwords, keep only alphabetic and remove words with length <3
        tokens = [word for word in tokens if word not in stop_words and word.isalpha() and len(word) >= 3]

        # 5. Lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return tokens

    # Apply to content column and create new column
    df['cleaned_content'] = df['content'].apply(clean_text)

    return df

#Read csv, preprocess it, save it as a new file
df = pd.read_csv(output_csv)
df_preprocessed = preprocess_content(df)

# Convert 'cleaned_content' column to JSON string before saving to CSV
df_preprocessed['cleaned_content'] = df_preprocessed['cleaned_content'].apply(json.dumps)
df_preprocessed.to_csv('nvda_preprocessed.csv', index=False)