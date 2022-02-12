#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:15:47 2020

@author: shruti
"""

import pandas as pd
import gensim
from gensim import corpora
from pprint import pprint
import spacy
import re     
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from gensim.models import CoherenceModel
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from Preprocess import Preprocess
from Viz import Viz
# !pip install -U tmtoolkit
# !python -m tmtoolkit setup


from tmtoolkit.topicmod.tm_gensim import evaluate_topic_models
from tmtoolkit.topicmod.evaluate import results_by_parameter

class LDA():
    
    def __init__(self, id_, corpus, num_topics, text, lda_model):
        
        self.id2word = id_
        self.corpus = corpus
        self.num_topics = num_topics
        self.stop = text
        
#         lda_model =  gensim.models.LdaModel(corpus=self.corpus,
#                                            id2word=self.id2word,
#                                            num_topics= self.num_topics, 
#                                            random_state=100,
#                                            chunksize=100,
#                                            passes=20,
#                                            per_word_topics=True)
        self.model = lda_model
        coherence_model_lda = CoherenceModel(model=lda_model, texts=self.stop, dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        self.coherence = coherence_lda
        print('\nCoherence Score: ', coherence_lda)
        
    #Compute funding per topic     
    def funding_per_topic(self, file):
        
        import numpy as np
        
        docs_top = self.dominant_topics_funds()
        
        raw_data = pd.read_csv(file, sep=',', encoding ='latin1')
        money = raw_data['AwardedAmountToDate']
        
        num_topics = self.num_topics
        money_ = {i: [] for i in range(self.num_topics)}
        
        for topicID, docs in docs_top.items():
            
            amt = 0
            d = docs_top[topicID][0]
            for mons in money[d]:
                amt = amt + int(re.sub(r'[^\w\s]', '', str(mons)))
            money_[topicID] = amt
            
        return money_
        
    def total_funding(self, file):
        
        import numpy as np
        
        raw_data = pd.read_csv(file, sep=',', encoding ='latin1')
        money = raw_data['AwardedAmountToDate']
        amt = 0
        for mons in money[d]:
            amt = amt + int(re.sub(r'[^\w\s]', '', str(mons)))

            
        return amt
        
    
    def dominant_topics(self, num_docs):
    
        topic_dict = {i: [] for i in range(self.num_topics)} 
        topic_dict
        for docID in range(len(self.corpus)):
            if (docID % 100 == 0): print(docID)
            topic_vector = self.model[self.corpus[docID]]
            for topicID, prob in topic_vector:
                topic_dict[topicID].append([docID, prob])

        docs_top_20 = {i: [] for i in range(self.num_topics)} 
        for topicID, doc_probs in topic_dict.items():
            doc_probs = sorted(doc_probs, key = lambda x: x[1], reverse = True)
            docs = [dp[0] for dp in doc_probs[:num_docs]]
            docs_top_20[topicID].append(docs)

        return docs_top_20
    
    def dominant_topics_funds(self):

        topic_dict = {i: [] for i in range(self.num_topics)} 
        topic_dict
        for docID in range(len(self.corpus)):
            if (docID % 100 == 0): print(docID)
            topic_vector = self.model[self.corpus[docID]]
            for topicID, prob in topic_vector:
                topic_dict[topicID].append([docID, prob])

        docs_top_20 = {i: [] for i in range(self.num_topics)} 
        for topicID, doc_probs in topic_dict.items():
            doc_probs = sorted(doc_probs, key = lambda x: x[1], reverse = True)
            docs = [dp[0] for dp in doc_probs[:len(self.corpus)] if dp[1] >= 0.5]
            docs_top_20[topicID].append(docs)

        return docs_top_20
    
    
    def dominant_directorate(self, file):
    
        raw_data = pd.read_csv(file, sep=',', encoding ='latin1')
        direc = raw_data['NSFDirectorate']
        
        dir_ = {i: {} for i in range(self.num_topics)} 
        docs_top = self.dominant_topics_funds()
        for topicID, docs in docs_top.items():
            
            d = docs_top[topicID][0]
            
            for doc in d:
                directorate = direc[doc]
                if directorate in dir_[topicID]:
                        dir_[topicID][directorate] = dir_[topicID][directorate] + 1
                else: 
                         dir_[topicID][directorate] = 1
                    

        return dir_
    
    def dominant_org(self, file):
    
        raw_data = pd.read_csv(file, sep=',', encoding ='latin1')
        direc = raw_data['NSFOrganization']
        
        dir_ = {i: {} for i in range(self.num_topics)} 
        docs_top = self.dominant_topics(len(self.corpus))
        for topicID, docs in docs_top.items():
            
            d = docs_top[topicID][0]
            
            for doc in d:
                directorate = direc[doc]
                if directorate in dir_[topicID]:
                        dir_[topicID][directorate] = dir_[topicID][directorate] + 1
                else: 
                         dir_[topicID][directorate] = 1
                    

        return dir_
    
    def topic_wordcloud(self, n, plot=False):
        
        viz = Viz(self.model, self.corpus)
        topics = viz.top_n_wordcloud(n, self.num_topics, plot)
        return topics
    
    def evaluate(self, top_l, top_h, step):

        var_params = [{'n_topics': k, 'alpha': 1/k} for k in range(top_l, top_h, step)]
        const_params = {'n_iter': 1000, 'eta': 0.1,'random_state': 20191122}  # to make results reproducible}
        eval_results = evaluate_topic_models(self.id2word,
                                     varying_parameters=var_params,
                                     constant_parameters=const_params,
                                     return_models=True)
        
        eval_results_by_topics = results_by_parameter(eval_results, 'n_topics')
        
        from tmtoolkit.topicmod.visualize import plot_eval_results

        plot_eval_results(eval_results_by_topics);
