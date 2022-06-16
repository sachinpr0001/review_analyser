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
from model import get_review
app = Flask(__name__)



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



@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET','POST'])
def csv_to_df():
    if request.method == 'POST':
        f = request.form['csvfile']
        data=[]
        with open(f, encoding = 'utf8') as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                data.append(row)    
        print(type(data))
        df = pd.DataFrame(data)
        df.to_csv('sample.csv',header=False, index = False) 
        df = pd.read_csv('sample.csv')
        review = get_review(df)
        return render_template('data.html', data=review.to_html(header=True,index=False))     

if __name__ == '__main__':
    app.run(debug=True)    

