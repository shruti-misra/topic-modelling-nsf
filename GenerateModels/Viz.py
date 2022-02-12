#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:57:48 2020
https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
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
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

class Viz():
    
    def __init__(self, model, corpus):
        
        self.model = model
        self.corpus = corpus
    
    def LDA_viz(self):
        
        import pyLDAvis.gensim
        
        lda_model = self.model
        corpus = self.corpus
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
        vis
        
    def tsne_viz(self):
        
        # Get topic weights and dominant topics ------------
        from sklearn.manifold import TSNE
        from bokeh.plotting import figure, output_file, show
        from bokeh.models import Label
        from bokeh.io import output_notebook
        
        lda_model = self.model
        corpus = self.corpus
        # Get topic weights
        topic_weights = []
        for i, row_list in enumerate(lda_model[corpus]):
            topic_weights.append([w for i, w in row_list[0]])
        
        # Array of topic weights    
        arr = pd.DataFrame(topic_weights).fillna(0).values
        
        # Keep the well separated points (optional)
        arr = arr[np.amax(arr, axis=1) > 0.35]
        
        # Dominant topic number in each doc
        topic_num = np.argmax(arr, axis=1)
        
        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(arr)
        
        # Plot the Topic Clusters using Bokeh
        output_notebook()
        n_topics = 4
        mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
        plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
                      plot_width=900, plot_height=700)
        plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
        show(plot)
        
        
        
    def top_n_wordcloud(self, n, topic_n, plot=False):
        
        
        from matplotlib import pyplot as plt
        from wordcloud import WordCloud
        import matplotlib.colors as mcolors
        
        
        lda_model = self.model
        
        #cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
        
        cloud = WordCloud(background_color='white',
                          width=2500,
                          height=1800,
                          max_words=n,
                          colormap='tab20',
                          #color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topics = lda_model.show_topics(formatted=False, num_topics=topic_n)
        print(len(topics))
        #fig, axes = plt.subplots(4,4, figsize=(10,10), sharex=True, sharey=True)
        
        if plot: 
            for i in range(0, len(topics)):

                topic_words = dict(topics[i][1])
                cloud.generate_from_frequencies(topic_words, max_font_size=250)
                plt.gca().imshow(cloud)
                plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
                plt.gca().axis('off')
                plt.axis('off')
                plt.margins(x=0, y=0)
                plt.tight_layout()
                plt.show()
        
        #plt.subplots_adjust(wspace=0, hspace=0)

        return topics
        
    
    def dominant_topics(self, plot_doc):
        
        
        lda_model = self.model
        corpus = self.corpus
        # Sentence Coloring of N Sentences
        def topics_per_document(model, corpus, start=0, end=1):
            corpus_sel = corpus[start:end]
            dominant_topics = []
            topic_percentages = []
            for i, corp in enumerate(corpus_sel):
                topic_percs, wordid_topics, wordid_phivalues = model[corp]
                dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
                dominant_topics.append((i, dominant_topic))
                topic_percentages.append(topic_percs)
            return(dominant_topics, topic_percentages)
        
        dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            
        
        # Distribution of Dominant Topics in Each Document
        df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
        dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
        df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()
        
        # Total Topic Distribution by actual weight
        topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
        df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()
        
        # Top 3 Keywords for each Topic
        topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
                                         for j, (topic, wt) in enumerate(topics) if j < 3]
        
        df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
        df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
        df_top3words.reset_index(level=0,inplace=True)
        
        if plot_doc:
            from matplotlib.ticker import FuncFormatter

        # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)

            # Topic Distribution by Dominant Topics
            ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
            ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
            tick_formatter = FuncFormatter(lambda x, pos:  str(x)+ '\n')
            ax1.xaxis.set_major_formatter(tick_formatter)
            ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
            ax1.set_ylabel('Number of Documents')
            ax1.set_ylim(0, 8000)

            # Topic Distribution by Topic Weights
            ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
            ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
            ax2.xaxis.set_major_formatter(tick_formatter)
            ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))

            plt.show()
        
        return dominant_topics, topic_percentages