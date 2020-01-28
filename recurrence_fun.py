# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:27:30 2020

@author: Vsolanki
"""

import pandas as pd
from flask import Flask, request,jsonify
from flask import Flask

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
def preprocess(sent_list):
        stoplist = list(string.punctuation)
        stoplist += stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        final_list = []
        for i in sent_list:
            review = re.sub(r'^br$', ' ', str(i))
            review = review.lower()
            review = re.sub(r'\s+br\s+', ' ', review)
            review = re.sub(r'[^\w\s]', ' ', review)
            review = re.sub(r'_', ' ', review)
            review = re.sub(r'\s+[a-z]\s+', ' ', review)
            review = re.sub(r'^b\s+', '', review)
            review = re.sub(r'\s+', ' ', review)
            review = re.sub(r'\d+', '', review)
            words = nltk.word_tokenize(review)
            words = [word for word in words if word not in stoplist]
            words = [lemmatizer.lemmatize(word) for word in words]
            final = " ".join(words)
            final_list.append(final)
    
        final_list = [x for x in final_list if str(x) != 'nan']
        return final_list   
    
def similarity(final_list):
    
    global tfidf
    tfidf = TfidfVectorizer(max_features=7000, stop_words='english')
    sparse_matrix = tfidf.fit_transform(final_list)

    doc_term_matrix = sparse_matrix.todense()
    '''df = pd.DataFrame(doc_term_matrix, 
                  columns=tfidf.get_feature_names())'''

    # Compute Cosine Similarity
    df_similar = pd.DataFrame(cosine_similarity(doc_term_matrix, doc_term_matrix))
    df_similar.columns = [str(i) for i in range(df_similar.shape[1])]
    df_similar.index = [str(i) for i in range(df_similar.shape[0])]

    np.fill_diagonal(df_similar.values, 0)

    R, C = np.where(df_similar.values > 0.6)

    df_out = pd.DataFrame(np.column_stack((df_similar.index[R], df_similar.columns[C], df_similar.values[R, C])),
                          columns=[['Sentence_num', 'Count', 'C']])

    df_out.columns = df_out.columns.get_level_values(0)
    df_out = df_out.sort_values('C', ascending=False).drop_duplicates(['Count'])
    df_out = df_out.reset_index()
    df_out.drop('C', axis=1, inplace=True)
    df_out.drop('index', axis=1, inplace=True)
    df_final = df_out.groupby('Sentence_num')['Count'].nunique()
    df_final = df_final.reset_index()
    group =df_out.groupby('Sentence_num')['Count'].unique()
    df_final['sen_index']=[list(x) for x in group]
    mapped_list = []
    for j in df_final['Sentence_num']:
        #blank.append(Final_filter["Date"][int(j)])
        mapped_list.append(sent_list[int(j)])

    df_final['Sentence'] = mapped_list

    imp_list = []
    for j in df_final['Sentence_num']:
        #blank.append(Final_filter["Date"][int(j)])
        imp_list.append(final_list[int(j)])

    df_final['Short description'] = imp_list
    
    return df_final


def recurrence(data):
    data = data[data['Short description'].notnull()].reset_index()
    data.reset_index(inplace=True)
    global sent_list
    sent_list = data['Short description'].values.tolist()[:20000]
    
    final_list = preprocess(sent_list)
    final_merge = similarity(final_list)
    df_latest = final_merge.sort_values('Count', ascending=False)[:500]
    
    df_latest.reset_index(inplace = True)
    df_latest.drop('index', axis = 1, inplace = True)
    sparse_matrix_top = tfidf.transform(df_latest['Short description'].values.tolist())
    
    doc_term_matrix_top = sparse_matrix_top.todense()
    top_similarity = cosine_similarity(doc_term_matrix_top, doc_term_matrix_top)
    
    np.fill_diagonal(top_similarity, 0)
    
    Row, Column = np.where(top_similarity > 0.5)
    
    df_out = pd.DataFrame(np.column_stack((Row, Column, top_similarity[Row, Column])),
                          columns=[['Sentence_num', 'Count', 'Value']])
    df_out.columns = df_out.columns.get_level_values(0)
    df_out['Value']= df_out['Value'].apply(lambda x: round(x, 2))
    
    s = []                
    for i in range(len(df_out)):
            for j in range(i,len(df_out)):
                if (df_out['Count'][i] == df_out['Sentence_num'][j]):
                    s.append(df_out.index[j])
    for i in set(s):
        df_out.drop(i, inplace = True)
    
    df_out = df_out.reset_index()
    df_out.drop('Value', axis=1, inplace=True)
    df_out.drop('index', axis=1, inplace=True)      
    df_final = df_out.groupby('Sentence_num')['Count'].nunique()
    df_final = df_final.reset_index()
    
    group =df_out.groupby('Sentence_num')['Count'].unique()
    df_final['sen_index']=[list(x) for x in group]
    remove=[]
    for i in (range(len(df_final))):
        for j in df_final.sen_index[int(i)]:
            remove.append(int(j))
            df_latest.sen_index[df_final.Sentence_num[int(i)]].extend(df_latest.sen_index[int(j)])
            df_latest.Count[df_final.Sentence_num[int(i)]] = df_latest.Count[df_final.Sentence_num[int(i)]]  + df_latest.Count[j]          
    df_latest.drop(df_latest.index[remove], inplace=True)  
    df_latest = df_latest[:100]
    #df_latest = df_latest.loc[df_latest["Count"]>=10]
    df_latest = df_latest.reset_index()  
    data['Created'] = pd.to_datetime(data['Created'])
    data['Created'] = data['Created'].dt.date 
    Date=[]       
    for i in range(len(df_latest)):
        temp_date=[]
        for j in df_latest.sen_index[int(i)]:
            temp_date.append(data.Created[int(j)])  #Name of date column should be "Created"
        Date.append(temp_date)
    df_latest["Date"]=Date
    Date_recent=[]       
    for i in range(len(df_latest)):
        temp_date=[]
        for j in df_latest.sen_index[int(i)]:
            temp_date.append(data.Created[int(j)])
            
            temp_date=list(set(temp_date))
            temp_date = sorted(temp_date,reverse=True)
            #Name of date column should be "Created"
        Date_recent.append(temp_date[0:5])
    df_latest["Recent Date"]=Date_recent
    global dataframe_recurrence
    dataframe_recurrence = df_latest[['Sentence','Count','sen_index','Recent Date', 'Date']]
    dataframe_recurrence = dataframe_recurrence.sort_values('Count', ascending=False)
    dataframe_recurrence.reset_index(inplace = True)
    dataframe_recurrence.drop('index', axis = 1, inplace = True)
    return dataframe_recurrence


def similar_search(input_sentence):
    input_sentence = [input_sentence]
    input_sentence = preprocess(input_sentence)
    sparse_matrix_input = tfidf.transform(input_sentence).todense()
    df_matching = preprocess(dataframe_recurrence['Sentence'].values.tolist())
    sparse_matrix_match = tfidf.transform(df_matching).todense()
    top_similarity_match = []
    matching_input = cosine_similarity(sparse_matrix_match,sparse_matrix_input)

    dataframe_recurrence['Matches'] = matching_input
    dataframe_recurrence1 = dataframe_recurrence.loc[dataframe_recurrence["Matches"]>0]
    if len(dataframe_recurrence1) != 0:
        dataframe_output = dataframe_recurrence1.sort_values('Matches', ascending=False)[:3]
    else:
        print("No match found for input problem")
    dataframe_output.reset_index(inplace = True)
    dataframe_output.drop('index', axis = 1, inplace = True)
    return dataframe_output

