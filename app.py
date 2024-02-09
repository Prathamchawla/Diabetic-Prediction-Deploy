#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from scipy.stats import zscore
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# In[2]:


app = Flask(__name__)


# In[4]:


# Load scaler and model
with open('diabeticscaler.pkl', 'rb') as scaler_file:
    diabitic_scaler = pickle.load(scaler_file)

with open('diabeticmodel.pkl', 'rb') as model_file:
    diabitic_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('diabeticmodel.html')

@app.route('/diabiticmodelpredict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        scaled_features = diabitic_scaler.transform([features])
        prediction = diabitic_model.predict(scaled_features)

        result = None
        if prediction[0] == 1:
            result = 'The patient is diabetic.'
        else:
            result = 'The patient is not diabetic.'

        return render_template('diabeticmodel.html', prediction=result)


# In[ ]:


if __name__ == '__main__':
    app.run(host='0.0.0.0')

