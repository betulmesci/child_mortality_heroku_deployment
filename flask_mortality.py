# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 19:13:14 2022

@author: h
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    prediction = model.predict(final)
    output = round(prediction[0], 2)
    return render_template('index.html',prediction_text='Child mortality rate is {}'.format(output))
    
if __name__ =='__main__':
    app.run(port=5000,debug=True)
