from flask import Flask, render_template, request, jsonify
# from model_api import make_prediction
from module.model import *

app = Flask(__name__)
# load data
rv = ReviewData()

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
    rvs = rv.review_df
    print(rvs[rvs['user_id']==int(x)])

    # es_predict = es.predict(request.args['input'])
    return render_template('predict.html', title='Predict', choice = x) #, id_input=es_predict)

# @app.route('/predict',methods=['POST'])
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)

#     output = round(prediction[0], 2)

#     return render_template('predict.html', prediction_text='Employee Salary should be $ {}'.format(output))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)


if __name__ == '__main__':
    app.run(debug = True)
    
# '''
# A Web application that shows Google Maps around schools, using
# the Flask framework, and the Google Maps API.
# '''

# from flask import Flask, render_template, abort
# app = Flask(__name__)


# class School:
#     def __init__(self, key, name, lat, lng):
#         self.key  = key
#         self.name = name
#         self.lat  = lat
#         self.lng  = lng

# schools = (
#     School('hv',      'Happy Valley Elementary',   37.9045286, -122.1445772),
#     School('stanley', 'Stanley Middle',            37.8884474, -122.1155922),
#     School('wci',     'Walnut Creek Intermediate', 37.9093673, -122.0580063)
# )
# schools_by_key = {school.key: school for school in schools}


# @app.route("/")
# def index():
#     return render_template('index.html', schools=schools)


# @app.route("/<school_code>")
# def show_school(school_code):
#     school = schools_by_key.get(school_code)
#     if school:
#         return render_template('map.html', school=school)
#     else:
#         abort(404)

# app.run(host='localhost', debug=True)