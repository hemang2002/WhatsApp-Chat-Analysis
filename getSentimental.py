# -*- coding: utf-8 -*-
"""
Created on Tue May 28 18:12:14 2024

@author: hhmso
"""

import pandas as pd
import re
import emoji
import datetime 
from tqdm import tqdm

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

class getAnalysis:

    
    def __init__(self):
        
        self.stop_words, self.lemmatizer = self.get_stopwords_lemma()
        self.model, self.tokenizer = self.getModel()        

        
    def get_stopwords_lemma(self):
        
        lemmatizer = WordNetLemmatizer()
        languages = ['arabic', 'azerbaijani', 'danish', 'dutch', 'english', 'finnish', 
                     'french', 'german', 'greek', 'hungarian', 'indonesian', 'italian', 
                     'kazakh', 'nepali', 'norwegian', 'portuguese', 'romanian', 'russian', 
                     'slovene', 'spanish', 'swedish', 'turkish']
        stop_words = [] 
        for lang in languages:
            stop_words.extend(stopwords.words(lang))
            
        return stop_words, lemmatizer
    
    
    def getModel(self):
        
        model_name = "bhadresh-savani/bert-base-go-emotion"
        model = BertForSequenceClassification.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        return model, tokenizer
        
    
    def date_time(self, s):
        
        pattern='^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -'
        result = re.search(pattern, s)
        if result:
            return True
        return False 
    
    
    def getUpdatedText(self, text):
        
        messages = []
        only_mess = []
        only_mess_df = []
        
        for i in text:
            i = i.replace('\u202F', '')
            if not self.date_time(i):
                messages[-1] = messages[-1] + " " + i    
            else:
                messages.append(i)
    
        for message in tqdm(messages):
            
            date = message.split(" -")[0].strip()
            date = datetime.datetime.strptime(date, "%m/%d/%y, %I:%M%p")
            message = "".join(message.split(" -")[1:]).strip()
            sender = message.split(": ")
            
            if len(sender) > 1 and not re.match("<Media omitted>", message):
                
                sender = sender[0]
                
                message = "".join(message.split(": ")[1:]).strip()
                message = self.clean_text(message)
                
                mood = self.predict_emotion(message.strip())
                
                only_mess.append({sender.strip(): {"date": date, "message": message, "emotion": mood}})
                only_mess_df.append([sender.strip(), date, message, mood])

        return only_mess, pd.DataFrame(only_mess_df, columns = ["sender", "date", "message", "mood"])
                
    
    def get_wordnet_pos(self, tag):
        
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        
        
    def clean_text(self, message):
            
        # message = re.sub(r'\d+', '', message)
        message = message.replace("<Media omitted>", "")
        message = emoji.replace_emoji(message, replace='')
        message = re.sub(r'[^\w\s]', '', message)
        tokens = word_tokenize(message)
        
        tokens = [word for word in tokens if word not in self.stop_words]
        
        tagged_tokens = pos_tag(tokens)
        lemmatized_tokens = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag)) for word, tag in tagged_tokens]
        
        return ' '.join(lemmatized_tokens)
               
     
    def predict_emotion(self, message):
        
        emotions = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
            "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude",
            "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral"
        ]
    
        inputs = self.tokenizer(message, return_tensors = "pt", truncation = True, padding = True, 
                           max_length = 512)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        
        predicted_emotions = emotions[predicted_class_id]
        
        return predicted_emotions

# with open(r"C:\Users\hhmso\Desktop\Hemang\Project\Masters\Projects\WhatsApp_senti\Data\WhatsApp Chat with SESIT. NFSU.txt", encoding = "utf-8") as f:
#     text = f.readlines()
#     f.close()
# del f
# _, df = getAnalysis().getUpdatedText(text)
# df = pd.DataFrame(df, columns = ["sender", "date", "message", "mood"])
