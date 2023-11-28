import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from multiprocessing import Pool
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
from tqdm import tqdm
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in 
              stopwords.words('english')]

    return ' '.join(tokens)


def preprocess_texts_parallel(your_array, num_processes=4):
    with Pool(num_processes) as p:
        result = p.map(preprocess_text, your_array.ravel())
    return result


# Ensure consistent results from langdetect
DetectorFactory.seed = 0


# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'


def is_not_string(x):
    return not isinstance(x, str)


def process_in_batches(dataframe, function, batch_size=100):
    # Create an empty series to store results
    results = pd.Series(dtype=object)

    # Process in batches
    for start in tqdm(range(0, len(dataframe), batch_size)):
        end = start + batch_size
        batch = dataframe['review_text'][start:end]
        batch_results = batch.apply(function)
        results = results.append(batch_results)

    return results
