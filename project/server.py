import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib as jb

from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)

@app.route('/')
def index():
    df = pd.read_csv("data\Student_Performance.csv")
    df_html =df.head(5).to_html()
    return render_template('index.html',table_df=df_html)

@app.route('/predict',methods=['GET','POST'])
def prediction():
    model = jb.load('data/regression.joblib')
    if request.method == 'POST':
        hours = int(request.form['hours'])
        scores = int(request.form['scores'])
        activites = int(request.form['activities'])
        sleep = int(request.form['sleep'])
        papers = int(request.form['papers'])
        inputs = np.array([hours,scores,activites,sleep,papers])
        inputs = inputs.reshape(1,-1)
        prediction = model.predict(inputs)

        return render_template('index.html',prediction=prediction)
    else:
       return "No data"
    

if __name__ == '__main__':
    app.run(debug=True)