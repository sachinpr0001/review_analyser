from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import csv
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

def get_review(df):
    lemmatizer = WordNetLemmatizer()
    data = df
    # data = pd.read_csv('sample.csv')
    s_w = set(stopwords.words('english'))
    s_w.remove('not')
    s_w.remove('no')
    New_Review = []
    for r in data['Text']:
        r = re.sub(r'[^\w\s]', '', str(r))
        r = re.sub(r'\d','',r)
        r_t = word_tokenize(r.lower().strip())
        r_w_s = []
        for t in r_t:
            if t not in s_w:
                t = lemmatizer.lemmatize(t)
                r_w_s.append(t)
        c_r = " ".join(r_w_s)
        New_Review.append(c_r)
    data['New_Text'] = New_Review
    Less_or_equal_to_3 = data[data.Star <= 3]
    modd = SentimentIntensityAnalyzer()
    S_l = []
    for i in Less_or_equal_to_3['New_Text']:
        score = modd.polarity_scores(i)
        b_score = TextBlob(i).sentiment.polarity
        if (score['pos'] >= 0.7):
            S_l.append('+')
        else:
            S_l.append('-')

    Less_or_equal_to_3["sentiment"] = S_l

    Positive_Review_with_3_or_less = Less_or_equal_to_3[Less_or_equal_to_3.sentiment == '+']
    Positive_Review_with_3_or_less.drop('New_Text', axis = 1,inplace=True)
    Positive_Review_with_3_or_less.to_csv('final_output.csv')
    return Positive_Review_with_3_or_less

# def get_review(df):
#     lemmatizer = WordNetLemmatizer()
#     all_reviews = df
#     print(all_reviews.head())

#     sw_clean = set(stopwords.words('english'))
#     sw_clean.remove('not')
#     sw_clean.remove('no')
#     new_review = []
#     for review in all_reviews['Text']:
#         review = re.sub(r'[^\w\s]', '', str(review))
#         review = re.sub(r'\d','',review)
#         review_token = word_tokenize(review.lower().strip())
#         r_without_stop = []
#         for word in review_token:
#             if word not in sw_clean:
#                 word = lemmatizer.lemmatize(word)
#                 r_without_stop.append(word)
#         clean_review = " ".join(r_without_stop)
#         new_review.append(clean_review)
#     all_reviews['New_Text'] = new_review
#     req_review = all_reviews[all_reviews.Star <= 3]
#     sentiment_model = SentimentIntensityAnalyzer()
#     senti_list = []
#     for review in req_review['New_Text']:
#         score = sentiment_model.polarity_scores(review)
#         b_score = TextBlob(review).sentiment.polarity
#         if (score['pos'] >= 0.7):
#             senti_list.append('+')
#         else:
#             senti_list.append('-')      
#     req_review["sentiment"] = senti_list        
#     final_review = req_review[req_review.sentiment == '+']
#     final_review.drop('New_Text', axis = 1,inplace=True)
#     return final_review