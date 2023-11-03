         #                       Dania Jawad Lab11 20i0569
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify,render_template
import numpy as np
import requests
from flask import render_template
# from myapp import app

# Load data
# data = pd.read_csv('price-prediction.csv')

# # Split data into training and testing sets
# X_train = data[['PLOTS', 'PRICE']]
# y_train = data['PLOTS']
# X_test = data[['PLOTS', 'PRICE']]
# y_test = data['PLOTS']

# # Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

app = Flask(__name__)

# Save the model
# with open('linear_regression.pkl','wb') as f:
#     pickle.dump(['PLOTS','PRICE'],f)

# @app.route('/')
# def index():
#     title = 'Linear Regression!'
#     heading = 'plot size:'
#     items = ['Item 1', 'Item 2', 'Item 3']
#     return render_template('form.html', title=title, heading=heading, items=items)


# with open('linear_regression.pkl','wb') as f:
#     pickle.dump(['PLOTS','PRICE'],f)

# Define a Flask endpoint
@app.route('/',methods=['GET','POST'])
def predict():
    # with open('linear_regression.pkl','rb') as f:
    # #  m,b = pickle.dump(['PLOTS','PRICE'],f)
    # model = pickle.load(open('linear_regression.pkl', 'rb'))

    # with open('linear_regression.pkl','rb') as f:
    #  m,b = pickle.dump(['PLOTS','PRICE'],f)
    # model = pickle.load(open('linear_regression.pkl', 'rb'))
    # data = request.get_json()
    # X = np.array([data['PLOTS'],data['PRICE']])
    # prediction = model.predict(X.reshape(1, -1))[0]
    # return jsonify({'prediction':prediction})
    global m,b
    if request.method == 'POST':
        plot_size = float(request.form['plot_size'])
        prediction = b+plot_size*m
        return render_template('form.html',prediction=prediction)
    else: 
        return render_template('form.html')
                       
if __name__ == '__main__':
    m=0
    b=0
    with open('model.pkl', 'rb') as f:
        m, b = pickle.load(f)

    app.run(debug=True)


