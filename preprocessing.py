import pandas as pd
import os
import spacy  # nlp
import regex as re
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from symspellpy import SymSpell, Verbosity

os.system('python -m spacy download en_core_web_sm')

nlp = spacy.load('en_core_web_sm')

# get SBERT model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

sym_spell = SymSpell()
dictionary_path = "./data/frequency_dictionary_en_82_765.txt.txt"
sym_spell.load_dictionary(dictionary_path, 0, 1)


def to_lower(x):
    return str.lower(x)


def strip_all_entities(x):
    entity_prefixes = ['@', '#']
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            x = x.replace(separator, ' ')
    words = []
    for word in x.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


def remove_tags(x):
    return re.sub(r"(?:\@|http?\://|https?\://|www)\S+", ' ', x)


def remove_punc(x):
    return re.sub(r'[^\w\s]', ' ', x)


def spelling_correction(term):
    doc_w_correct_spelling = []
    for tok in term.split(" "):
        x = sym_spell.lookup(tok, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)[0].__str__()
        y = x.split(',')[0]
        doc_w_correct_spelling.append(y)

    return " ".join(doc_w_correct_spelling)


def extract_keywords(nlp=nlp, doc="", no_of_keywords=5, model=model):

    doc = doc.lower()
    doc = re.sub(r'(?:\@|http?\://|https?\://|www)\S+', ' ', doc)
    doc = re.sub(r'[^\w\s]', ' ', doc)
    doc = re.sub(' \d+', ' ', doc)

    doc_ = nlp(doc)

    pos_tag = ['VERB', 'NOUN', 'ADJ', 'PROPN']
    result = []

    for token in doc_:
        if token.pos_ in pos_tag:
            result.append(token.text)

    doc_embedding = model.encode([doc])
    results_embeddings = model.encode(result)

    try:
        distances = cosine_similarity(doc_embedding, results_embeddings)
        keywords = [result[index] for index in distances.argsort()[0][-no_of_keywords:]]
    except:
        return "NaN"

    return keywords


def get_location(nlp=nlp, doc=""):
    doc_ = nlp(doc)

    location = ""

    for ent in doc_.ents:
        if ent.label_ in ["GPE", "ORG"]:
            location = location + ent.text + "NaN"

    return location


def lemmatize(sentence):
    doc = nlp(sentence)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)


def remove_stopwords(sentence):
    doc = nlp(sentence)
    all_stopwords = nlp.Defaults.stop_words
    doc_tokens = [token.text for token in doc]
    tokens_without_sw = [word for word in doc_tokens if word not in all_stopwords]
    return " ".join(tokens_without_sw)
