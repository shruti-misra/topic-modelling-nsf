#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 23:36:40 2020

@author: shruti
"""

import pandas as pd
import gensim
from gensim import corpora
from pprint import pprint
import spacy
import re     
# Plotting tools
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from gensim.utils import simple_preprocess, lemmatize

class Preprocess():
    
    def __init__(self, file):
        
	#Takes data in as a csv file and converts it to a pandas data frame
        self.raw_data = pd.read_csv(file, sep=',', encoding ='latin1')

	#Extract the abstracts in the "Abstracts column of the proposal
        self.df_new1 = self.raw_data[self.raw_data['Abstract'].notnull()]

        
    def clean(self, text):
        clean_data = []
        for x in (text): #this is Df_pd for Df_np (text[:])
            new_text = re.sub('<.*?>', '', x)   # remove HTML tags
            new_text = re.sub(r'[^\w\s]', '', new_text) # remove punctuation
            new_text = re.sub(r'\d+','',new_text)# remove numbers
            new_text = new_text.lower() # lower case, .upper() for upper          
            if new_text != '':
                clean_data.append(new_text)
        return clean_data
        
    def clean_data(self):
        
        clean_abs = [self.clean(i.split()) for i in self.df_new1['Abstract']]
        #clean_title = [self.clean(i.split()) for i in self.df_new1['Title']]
        data = clean_abs #+ clean_title
        return data 
    
    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(self, texts, stop_words):
        stopped =  [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
        data_stop_joined = [' '.join(words) for words in stopped] 

        #from wordcloud import WordCloud
        # Create a WordCloud object
        #wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=4, contour_color='steelblue')
        # Generate a word cloud
        #wordcloud.generate(str(data_stop_joined))
        # Visualize the word cloud
        #img = wordcloud.to_image()
        #img.show()
        
        return stopped 
    
    def make_bigrams(self, data):
        
        common_terms = ["machine_learning", "computer_vision", "natural_language_processing", "neural_network",
                        "artificial_intelligence", "pattern_recognition", "computational_biology", "speech_recognition", 
                        "data_science", "big_data", "statistical_learning", "data_mining"]
        
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data, min_count= 2, threshold=3, common_terms = common_terms) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data], threshold=3, common_terms = common_terms)  
        
        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        
        return [bigram_mod[doc] for doc in data]
    
    def make_trigrams(self, texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    def lemmatization(self, texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    
    def lemmatize(self, data):
        
        # Form Bigrams
        data_words_bigrams = self.make_bigrams(data)
        
        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])
        
        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(data_words_bigrams,nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        data_lem_joined = [' '.join(words) for words in data_lemmatized] 

        #from wordcloud import WordCloud
        # Create a WordCloud object
        #wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=4, contour_color='steelblue')
        # Generate a word cloud
        #wordcloud.generate(str(data_lem_joined))
        # Visualize the word cloud
        #img = wordcloud.to_image()
        #img.show()
        
        return data_lemmatized

    def plot_most_common_words(self, data, n):
       
        data_lem_joined = [' '.join(words) for words in self.lemmatize(data)] 
        count_vectorizer = CountVectorizer(stop_words='english')
        count_data = count_vectorizer.fit_transform(data_lem_joined)
        words = count_vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))
        for t in count_data:
            total_counts+=t.toarray()[0]
        
        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:n]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words)) 
        
        print(words)
    
        plt.figure(2, figsize=(15, 15/1.6180))
        plt.subplot(title='10 most common words')
        sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
        sns.barplot(x_pos, counts, palette='husl')
        plt.xticks(x_pos, words, rotation=90) 
        plt.xlabel('words')
        plt.ylabel('counts')
        plt.show()
        
        return words

    def corpus(self, data_words_nostops):
        # Create Dictionary
        id2word = corpora.Dictionary(data_words_nostops)
        # Create Corpus
        texts = data_words_nostops
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        
        return id2word, corpus
        
    #Clean and lemmatize 
    def clean_lemmatize(self, plot, n):
        
        cleaned = self.clean_data()
        lemmed = self.lemmatize(cleaned)
        if (plot):
            words = self.plot_most_common_words(lemmed,n)
        else:
            words = []
        return lemmed, words
    
    #Create corpus
    def create_corpus(self, lem, stop_words):
        
        stop = self.remove_stopwords(lem, stop_words)
        id_, corpus = self.corpus(stop)
        return id_, corpus, stop
            
            
        

        
