# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:53:32 2015

@author: Amelie
"""

import os, sys, re, json, itertools, operator
import pandas as pd, numpy as np, scipy as sp
from joblib import Memory
from bson import json_util
    
#%load_ext cythonmagic       # '%' = magic function in ipython

data_directory = os.path.join('data')
output_directory = os.path.join(os.path.expanduser("~"), 'work', 'stockmeme', 'news_analysis')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

joblib_cache = Memory(cachedir=os.path.join(output_directory, 'joblib'), verbose=0)

pd.set_option('notebook_repr_html', True)
pd.set_option('precision', 4)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 50)
pd.set_option('max_colwidth', 500)
pd.set_option('column_space', 100)
pd.set_option('use_inf_as_null', True)

#!pwd

with open( os.path.join(data_directory, 'bloomberg.json') ) as f:
    bloomberg = json_util.loads(f.read())   #bloomberg = large dictionary (b/c json format)

df_bloomberg = pd.DataFrame(bloomberg)      #pandas can take json data & build data pandas 
df_bloomberg = df_bloomberg.drop('_id', axis=1); #drop 'name of colmumn'; axis = 1 specifies that '_id" is a column not a row (1 = row)
df_bloomberg = df_bloomberg.sort('insertion_date', ascending=False)
df_bloomberg = df_bloomberg.reset_index().drop('index', axis=1)
## drop, sort etc are pandas methods -- hence used on pandas obj 
df_bloomberg.head(10)

df_bloomberg.text[1]            #.text --> text = column header b/c of dot; indexing to first element

df_reuters.text[0]

"""
goals:
1. clean text
    a. steal useful regular expressions: http://ravikiranj.net/drupal/201205/code/machine-learning/how-build-twitter-sentiment-analyzer
2. run NLTK:
    a. find named entities
    b. find company names
    c. find topics
    d. synthesize adjectives
    e. cluster articles from both Reuters and Bloomberg, group similar articles
    f. get POS tags - what can I do with this?
3. AFTER 2 is complete
    a. look at bigrams for each text block
    b. take TF-IDF across documents
    c. find unique bigrams/trigrams
    d. count unique bigrams/trigrams across documents in discrete time-buckets
"""

import nltk
from nltk.tag.stanford import StanfordTagger
from nltk.collocations import *
from text.blob import TextBlob

articles = df_bloomberg.text.tolist()     #.tolist() --> changes type from str to list
text = '  '.join(sentence for sentence in articles[0])  #join sentences into 1 string sep by dbl space
#text = text.encode("utf-8")

def find_entities(chunks):
    "given list of tagged parts of speech, returns unique named entities"

    def traverse(tree):
        "recursively traverses an nltk.tree.Tree to find named entities"
        entity_names = []
    
        if hasattr(tree, 'node') and tree.node:
            print tree.node
            if tree.node == 'GPE':
                entity_names.append(' '.join([child[0] for child in tree]))
            elif tree.node == 'PERSON':
                entity_names.append(' '.join([child[0] for child in tree]))
            else:
                for child in tree:
                    entity_names.extend(traverse(child))
    
        return entity_names
    
    named_entities = []
    
    for chunk in chunks:
        entities = sorted(list(set([word for tree in chunk
                            for word in traverse(tree)])))
        for e in entities:
            if e not in named_entities:
                named_entities.append(e)
    return named_entities



 ## takes text and associate w. POS tags (so can filter on basis of tags)

tokens = nltk.word_tokenize(text)            
sentences = nltk.sent_tokenize(text)
words     = (nltk.word_tokenize(sentence) for sentence in sentences)
tags       = [nltk.pos_tag(word) for word in words]

named_entity_chunks = nltk.batch_ne_chunk(tags)
find_entities(named_entity_chunks)

# http://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity/12128777
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

sentences[:10]

tfidf = TfidfVectorizer(ngram_range=(1,3), token_pattern=r'\b\w+\b', min_df=1).fit_transform(cleaned_text) # need to use character n-grams

cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()

related_docs_indices = cosine_similarities.argsort()[:-20:-1]
related_docs_indices

cosine_similarities[related_docs_indices]

for i in related_docs_indices:
    print cleaned_text[i]

"""
Using cosine similarity as a distance function: we simply don't have enough characters to properly cluster tweets
"""

tfidf.data

vectorizer = CountVectorizer(min_df=1)

vectorizer.fit_transform([text[0]])

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_w


"""
Tag
Description
CC	Coordinating conjunction
CD	Cardinal number
DT	Determiner
EX	Existential there
FW	Foreign word
IN	Preposition or subordinating conjunction
JJ	Adjective
JJR	Adjective, comparative
JJS	Adjective, superlative
LS	List item marker
MD	Modal
NN	Noun, singular or mass
NNS	Noun, plural
NNP	Proper noun, singular
NNPS	Proper noun, plural
PDT	Predeterminer
POS	Possessive ending
PRP	Personal pronoun
PRP$	Possessive pronoun
RB	Adverb
RBR	Adverb, comparative
RBS	Adverb, superlative
RP	Particle
SYM	Symbol
TO	to
UH	Interjection
VB	Verb, base form
VBD	Verb, past tense
VBG	Verb, gerund or present participle
VBN	Verb, past participle
VBP	Verb, non-3rd person singular present
VBZ	Verb, 3rd person singular present
WDT	Wh-determiner
WP	Wh-pronoun
WP$	Possessive wh-pronoun
WRB	Wh-adverb
"""

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer as porter_stemmer

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
	tokens = nltk.word_tokenize(text)
	sentences = nltk.sent_tokenize(text)
	words = (nltk.word_tokenize(sentence) for sentence in sentences)
	tag_tuples = [nltk.pos_tag(word) for word in words]

	new_sentence = []
	for tuples in tag_tuples:
		for word, tag in tuples:
			if tag.startswith('V'):
				new_sentence.append( lemmatizer.lemmatize(word, wn.VERB) )
				
			elif tag.startswith('J'):
				new_sentence.append( lemmatizer.lemmatize(word, wn.ADJ) )
				
			elif tag.startswith('N'):
				new_sentence.append( lemmatizer.lemmatize(word, wn.NOUN) )
				
			elif tag.startswith('R'):
				new_sentence.append( lemmatizer.lemmatize(word, wn.ADV) )
				
			else:
				new_sentence.append( word )

	print text
	print ' '.join(i for i in new_sentence)

lemmatize_text(cleaned_text[5])

"""
Fun with the CMU tagger

https://github.com/brendano/ark-tweet-nlp
http://www.ark.cs.cmu.edu/TweetNLP/
https://github.com/ianozsvald/ark-tweet-nlp-python
"""

import CMUTweetTagger # will wrap this with a web-service

for text in cleaned_text[:10]:
    print text
    print CMUTweetTagger.runtagger_parse([text])
    print


#output:
#AT_USER $aapl. apple's iphone has cracked.
#[[('AT_USER', 'P', 0.5752), ('$aapl', '^', 0.7174), ('.', ',', 0.9668), ("apple's", 'Z', 0.6764), ('iphone', '^', 0.7309), ('has', 'V', 0.9833), ('cracked', 'V', 0.5413), ('.', ',', 0.9983)]]

#$aapl is holding well in the bull flag. did you notice the golden cross on the daily? ;) URL
#[[('$aapl', '^', 0.8645), ('is', 'V', 0.9961), ('holding', 'V', 0.9728), ('well', 'R', 0.8528), ('in', 'P', 0.9986), ('the', 'D', 0.9991), ('bull', 'N', 0.9745), ('flag', 'N', 0.9849), ('.', ',', 0.9979), ('did', 'V', 0.9994), ('you', 'O', 0.9957), ('notice', 'V', 0.9922), ('the', 'D', 0.999), ('golden', 'A', 0.4243), ('cross', 'N', 0.9899), ('on', 'P', 0.9987), ('the', 'D', 0.9991), ('daily', 'A', 0.5749), ('?', ',', 0.9897), (';)', 'E', 0.9774), ('URL', 'N', 0.4083)]]




"""
Interesting projects:
https://github.com/ianozsvald/social_media_brand_disambiguator
https://github.com/ianozsvald/twitter-text-python
https://github.com/ianozsvald/twitter_networkx_concept_map

http://blog.yhathq.com/posts/named-entities-in-law-and-order-using-nlp.html
http://blog.newsle.com/2013/02/01/text-classification-and-feature-hashing-sparse-matrix-vector-multiplication-in-cython/
https://textblob.readthedocs.org/en/latest/

"""










