import re
import unicodedata
import inflect
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


def remove_non_ascii(tokens):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in tokens:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(tokens):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in tokens:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(tokens):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in tokens:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(tokens):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in tokens:
        if word.isdigit() and int(word) < 40:  # keep numbers > 40 as they are
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(tokens):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in tokens:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(tokens):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in tokens:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(tokens):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in tokens:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(tokens):
    tokens = remove_non_ascii(tokens)
    tokens = to_lowercase(tokens)
    tokens = remove_punctuation(tokens)
    tokens = replace_numbers(tokens)
    tokens = remove_stopwords(tokens)

    return tokens
