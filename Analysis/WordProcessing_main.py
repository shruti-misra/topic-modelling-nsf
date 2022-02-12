import pandas as pd
import gensim
from gensim import corpora
from pprint import pprint
import spacy
import re     
import matplotlib.pyplot as plt
import pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from Preprocess import Preprocess
from wordcloud import WordCloud


#Choose file
f1 = 'Raw_data/Awards1415.csv' 

#Create Preprocess class with the file
p1= Preprocess(f1)

#Lemmatize- plot top 50 (n) words if set to True
lem, words = p1.clean_lemmatize(True, 50)

nltk.download('stopwords')
from nltk.corpus import stopwords

#Common English stopwords
stop_words = stopwords.words('english')

#Add custom stopwords
custom_stop = ['project', 'research', 'datum', 'use', 'learn', 'develop', 'model', 'new', 'student', 'application', 
              'algorithm', 'machine', 'provide', 'method', 'include', 'problem', 'science', 'information', 
              'technology', 'work', 'design', 'tool', 'analysis', 'technique', 'study', 'approach', 'support', 'also', 
              'result', 'development', 'improve', 'advance', 'learning', 'field', 'make', 'propose', 'area', 'enable', 
               'program', 'process', 'large', 'system', 'well', 'many']

stop_words = stop_words + custom_stop

words = ' '.join(word for word in sum(lem, []))

#Create WordCloud
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', 
                stopwords = stop_words,      
                min_font_size = 10).generate(words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()