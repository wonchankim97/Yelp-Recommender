from flask import Flask, render_template, request, jsonify
# from model_api import make_prediction
from module.model import *

app = Flask(__name__)
# load data
rv = ReviewData()
es = Ensemble(rv)

# # xlearn
# xl = PredictXLearn(rv)
# xl_predict = xl.predict()

# # deepfm
# dfm = PredictDeepFM(rv)
# dfm_predict = dfm.predict()

# # ensembled
# es = Ensemble(rv)

##############################

@app.route('/')
def running():
    return render_template('home.html')


@app.route('/demo', methods=['GET','POST'])
def demo():
    return render_template('demo.html', title='Demo')

@app.route('/demo/predict', methods=['GET','POST'])
def predict():
    # test
    # x = rv.review_df.head()
    # print(x)

    x = request.form['choice']
    #rvs = rv.review_df
    #print(rvs[rvs['user_id']==int(x)])
    
    es_predict = es.predict(x)
    es_predict_list = es_predict.iloc[:10].apply(lambda x : x.to_dict(), axis=1)

    es_predict2 = es.predict2(x)
    es_predict_list2 = es_predict2.iloc[:10].apply(lambda x : x.to_dict(), axis=1)

    user_for_cat = rv.restaurant_df[rv.restaurant_df['business_id'].isin(rv.review_df[rv.review_df['uid']==x]['bid'])]
    cat = []
    [cat.extend(n.split(',')) for n in user_for_cat['categories'] if n is not None ]
    cat = [n.strip() for n in cat]
    cat = [(n, cat.count(n)) for n in set(cat)]
    cat.sort(key=lambda e:e[1], reverse=True)
    user_cat = cat[:10]

    user = es._data_object.user_df[es._data_object.user_df['user_id']==x].iloc[0].to_dict()
    return render_template('predict.html', title='Predict', ranking = es_predict_list, ranking2 = es_predict_list2, user = user, user_cat=user_cat) #, id_input=es_predict)


if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0')