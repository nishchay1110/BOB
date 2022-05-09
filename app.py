import re
from flask import Flask, render_template, request, make_response
import pandas as pd
# import numpy as np
import nltk
nltk.download('punkt')
import pickle

app= Flask(__name__)
xgb_clf = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    file = request.form.get("qa", False)
    text_df=pd.read_excel(file)

    vectorizer_tf= pickle.load(open('vectorizer.pkl', 'rb'))
    full_vectors = vectorizer_tf.transform(text_df[' CUSTOMER_RESOLVER_COMM'].apply(str))

    predict_all = xgb_clf.predict(full_vectors)
    rev_catogeries = {0:'Proper',1:'Stereo Type Reply',2:'Futuristic',3:'Diverted',4:'Not Readable'}

    pred_df = pd.Series(predict_all).to_frame()
    pred_df.columns = ['Predicted']

    df_out = pd.merge(text_df,pred_df,how = 'left',left_index = True, right_index = True)

    df_out['Predicted CX Cat'] = df_out['Predicted'].map(rev_catogeries)

    df_out.drop(['Predicted'], axis = 1,inplace=True)

    df_out.to_csv('Data_predicted.csv', header=True, index=False)

    resp = make_response(df_out.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=Data_predicted.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


def tokenize(text):
    stemmer = nltk.stem.SnowballStemmer('english')
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in nltk.word_tokenize(text) if (len(word) > 3 and len(word.strip('www./'))>2 and len(re.sub('\d+','', word.strip('www./'))))]
    tokens = map(str.lower, tokens)
    stems = [stemmer. stem(item) for item in tokens if (item not in stop_words)]
    return stems
    

if __name__ == "__main__":
    app.run(debug=True)