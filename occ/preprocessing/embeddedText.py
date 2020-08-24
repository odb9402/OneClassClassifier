import tensorflow as tf
import tensorflow.keras.layers as tfkl
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
import numpy as np
import progressbar as pgb

class embeddedText():
    """
    Class for embedded documents.
    
    Attributes:
        vectors
        num_words
        latent_dim
        embedding_model
        word_vec
        
    Methods:
        embedding : texts, num_words, latent_dim = 
        
        (static) sentence_mean : sentence_arr = 
        
    """
    def __init__(self):
        self.word_vec = []
    
    def glove_embedding(self, texts, file):
        """
        Using pre-trained word-vector model, glove
        glove model can be downloaded from stanford nlp data repository:
        
        !wget http://nlp.stanford.edu/data/glove.6B.zip
        
        param: texts - Raw texts (sentence)
        param: file - The input glove file : glove.6B.*.txt
        
        return: None
        """
        self.embedding_dict = dict()
        glove_file = open(file, encoding='utf-8')
        for line in glove_file:
            word_vector = line.split()
            word = word_vector[0]
            word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
            self.embedding_dict[word] = word_vector_arr
        glove_file.close()
        
        i = 0
        with pgb.ProgressBar(max_value=len(texts)) as bar:
            for text in texts:
                vec = []
                text = text.split()
                for t in text:
                    try:
                        vec.append(self.embedding_dict[t.lower()])
                    except KeyError:
                        pass
                ## There are no matched words
                if len(vec) == 0:
                    print("len 0 vec")
                    self.word_vec.append(np.zeros((100)))
                else:
                    #print(np.array(vec))
                    #print(np.array(vec).shape)
                    sentence = self.sentence_vec(np.array(vec))
                    #print(sentence)
                    #print(sentence.shape)
                    self.word_vec.append(sentence)
                i += 1
                bar.update(i)
        self.word_vec = np.array(self.word_vec)
        print(self.word_vec.shape)
        
    def embedding(self, texts, num_words=None, latent_dim=16):
        """
        param: texts - A list of texts
        param: num_words - The number of dimension to use (sorted by frequencies).
        """
        self.__t = Tokenizer(num_words=num_words)
        self.__t.fit_on_texts(texts)
        self.vectors = list(map(lambda x:self.__t.texts_to_sequences([x]), texts))
        
        self.num_words = num_words
        self.latent_dim = latent_dim
        
        self.embedding_model = tfkl.Embedding(self.num_words, self.latent_dim)
        
        #i = 0
        #with pgb.ProgressBar(max_value=len(texts)) as bar:
        #    for v in self.vectors:
        #        self.word_vec.append(self.sentence_vec(self.embedding_model(np.array(v)))) 
        #        i += 1
        #        bar.update(i)
        #self.word_vec = np.array(self.word_vec)
        #s = self.word_vec.shape
        #self.word_vec = self.word_vec.reshape(s[0], s[-1])
        #self.word_vec = self.mask_nan(self.word_vec)
        
    @staticmethod
    def sentence_vec(sentence_arr):
        """
        param: sentence_arr - A numpy array with wordvectors
        """
        sentence_sum = np.sum(sentence_arr, axis=0)
        norm = np.linalg.norm(sentence_sum)
        return sentence_sum/norm 
    
    @staticmethod
    def sentence_mean(sentence_arr):
        """
        param: sentence_arr - A numpy array with wordvectors
        """
        return np.mean(sentence_arr, axis=1)
    
    @staticmethod
    def clean_text(text):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', text)
        cleantext = re.sub(r'[^\w]', ' ', cleantext) 
        return cleantext
    
    @staticmethod
    def mask_nan(array):
        return array[~np.isnan(array).any(axis=1)]
        
        
class amazoneText(embeddedText):
    """
    Class includes word-vectors of Amazon food reviews.
    
    Attributes:
        vectors
        num_words
        latent_dim
        embedding_model
        word_vec
    
    Methods:
        __init__ : file_name = 
    """
    def __init__(self, file_name):
        super().__init__()
        self.scores = []
        self.reviews = []
        self.raw_title = []
        self.raw_reviews = []
        self.raw_helpfulness = []
        
        file = open(file_name, 'r',  errors='ignore')
        file_str = file.readlines()
        
        review_string = ""
        i = 0
        with pgb.ProgressBar(max_value=len(file_str)) as bar:
            for line in file_str:
                line_splited = line.split(':')
                if line_splited[0] == 'review/score':
                    self.scores.append(float(line_splited[1]))
                elif line_splited[0] == 'review/summary':
                    self.raw_title.append(self.clean_html(line_splited[1].rstrip()))
                    review_string += self.clean_text(line_splited[1].rstrip()) 
                elif line_splited[0] == 'review/text':
                    self.raw_reviews.append(self.clean_html(line_splited[1].rstrip()))
                    review_string += self.clean_text(line_splited[1].rstrip()) 
                    self.reviews.append(review_string)
                    review_string = ""
                elif line_splited[0] == 'review/helpfulness':
                    self.raw_helpfulness.append(self.clean_html(line_splited[1].rstrip()))
                else:
                    pass
                i += 1
                bar.update(i)
        
        #self.glove_embedding(self.reviews)#, 50000, 32)
    
    @staticmethod
    def clean_html(text):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', text)
        return cleantext