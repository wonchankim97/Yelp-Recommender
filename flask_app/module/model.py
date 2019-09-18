import json
import re
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
from ast import literal_eval

import os
import plaidml.keras
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import xlearn as xl
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

### Load ##################
class ReviewData():
    def __init__(self):
        self.review_df = pd.DataFrame()
        self.restaurant_df = pd.DataFrame()
        self.restaurant_whole_df = pd.DataFrame()
        self.load_data()
    
    def load_data(self):
        df = pd.read_csv('review.csv',index_col='Unnamed: 0', converters={'categories': literal_eval})
        self.review_df = df
        df = pd.read_csv('restaurant.csv',index_col='Unnamed: 0')
        self.restaurant_df = df
        df = pd.read_csv('restaurant_whole.csv',index_col='Unnamed: 0')
        self.restaurant_whole_df = df

### PredictXLearn ##################
class PredictXLearn():
    def __init__(self, _data_object):
        self._user_id = ''
        self._xreviews = _data_object.review_df
        self._xrestaurants = _data_object.restaurant_df
        self._predict_df = pd.DataFrame()
        
        self.field_dict = {'Restaurant': '0', 'User': '1', 'Category': '2'}
        self.mapping_dict = {'useful': 'Restaurant', 'funny': 'Restaurant', 'cool':'Restaurant','review_count':'Restaurant',
                        'user_review_count': 'User', 'user_useful': 'User', 'user_funny':'User','user_cool':'User','fans':'User','average_stars':'User',
                        'categories': 'Category'}
        self.col_len_dict = {'useful':10,'funny':10,'cool':10,'review_count':10,'categories':88,'user_review_count':10,'user_useful':10,'user_funny':10,'user_cool':10,'fans':10,'average_stars':1}
        self.col_accum_index_dict = {}
        cumulative = 0
        for key, value in self.col_len_dict.items():
            self.col_accum_index_dict[key] = cumulative
            cumulative = cumulative + value
            
    def build_predict_df(self, _user_id):
        predict_df = self._xreviews[['useful','funny','cool','review_count','bid']]
        predict_df = predict_df.drop_duplicates(subset=['bid'])
        predict_user_df = self._xreviews[self._xreviews['uid']==self._user_id][['stars','user_review_count','user_useful','user_funny','user_cool','fans','average_stars','uid']]
        predict_user_df = predict_user_df.drop_duplicates(subset=['uid'])
        predict_df['key'] = 0
        predict_user_df['key'] = 0
        predict_df = predict_df.merge(predict_user_df, how='outer')
        self._predict_df = predict_df
        
    def build_libffm(self, _user_id):
        self._user_id = _user_id
        self.build_predict_df(_user_id)
        with open('xlearn_dataset/predict_'+self._user_id+'.txt', 'w') as txt_file:
            predict_df = self._predict_df
            for idx, row in predict_df.iterrows():
                vec = []
                label = row['stars']
                vec.append(str(label))
                row = row.drop(labels=['stars','bid','uid','key'])
                for key, value in row.items():
                    if key == 'average_stars':
                        col_idx = self.col_accum_index_dict[key]
                        out_val = value
                        vec.append(self.field_dict[self.mapping_dict[key]] + ":" + str(col_idx) + ":" + str(out_val))
                    elif key == 'categories':
                        col_idx = self.col_accum_index_dict[key] - 1
                        out_val = 1
                        [vec.append(self.field_dict[self.mapping_dict[key]] + ":" + str(col_idx+n) + ":" + str(out_val)) for n in value if n >0]
                    else:
                        col_idx = self.col_accum_index_dict[key] + (int(value) - 1)
                        out_val = 1
                        vec.append(self.field_dict[self.mapping_dict[key]] + ":" + str(col_idx) + ":" + str(out_val))
                txt_file.write("%s\n" % " ".join(vec))
                
    def predict(self, num=10):
        fm_model = xl.create_fm()
        # Prediction task
        predict_path = 'xlearn_dataset/predict_'+self._user_id+'.txt'
        fm_model.setTest(predict_path)  # Set the path of test dataset
        # Start to predict
        # The output result will be stored in output.txt
        fm_model.predict('final_model/xLearn_v0.1.out', 'xlearn_dataset/output_'+self._user_id+'.txt')
        result = pd.read_csv('xlearn_dataset/output_'+self._user_id+'.txt', header=None, names=['final_stars'])        
        result = pd.concat([self._xrestaurants,result], axis=1).sort_values('final_stars', ascending=False)[:num]

        return result
    
    def simple_predict(self):
        fm_model = xl.create_fm()
        # Prediction task
        predict_path = 'xlearn_dataset/predict_'+self._user_id+'.txt'
        fm_model.setTest(predict_path)  # Set the path of test dataset
        # Start to predict
        # The output result will be stored in output.txt
        fm_model.predict('final_model/xLearn_v0.1.out', 'xlearn_dataset/output_'+self._user_id+'.txt')
        result = pd.read_csv('xlearn_dataset/output_'+self._user_id+'.txt', header=None, names=['xlearn_stars'])        
        result = pd.concat([self._xrestaurants['business_id'],result], axis=1)

        return result


### DeepFM ##################
class PredictDeepFM():
    def __init__(self, _data_object, n_max_category = 12, n_category = 88):
        self._user_id = ''
        self._xreviews = _data_object.review_df
        self._xrestaurants = _data_object.restaurant_df
        self._xdf2xy = pd.DataFrame()
        self._predict_df = pd.DataFrame()
        self._n_max_category = n_max_category
        self._n_category = n_category
        self._deep_fm_model = load_model('final_model/deepFM_v0.1.h5')
    
    def build_predict_df(self, _user_id):
        final_k = pd.DataFrame({'business_id':self._xreviews['business_id'].unique()})
        final_review = final_k.merge(self._xreviews[['business_id','name','bid','restaurant_stars','review_count','categories']], how='left',on='business_id')
        final_review = final_review.drop_duplicates(subset=['business_id'])
        final_user = self._xreviews[self._xreviews['uid']==_user_id][['user_id','user_review_count','user_useful','user_funny','user_cool','fans','average_stars','stars']]
        final_user = final_user.drop_duplicates(subset=['user_id'])
        final_review['key'] = 0
        final_user['key'] = 0
        final_df = final_review.merge(final_user, how='outer').drop(['key'],axis=1)
        self._predict_df = final_df
        
    def build_df2xy(self, _user_id):
        self._user_id = _user_id
        self.build_predict_df(_user_id)
        self._predict_df = [self._predict_df.restaurant_stars.values,
                       self._predict_df.review_count.values,
                       self._predict_df.average_stars.values,
                       self._predict_df.user_id.values, 
                       self._predict_df.business_id.values, 
                       np.concatenate(self._predict_df.categories.values).reshape(-1,self._n_max_category)]
        
    def predict(self, num=10):
        result = pd.DataFrame(self._deep_fm_model.predict(self._predict_df), columns=['final_stars'])
        result = pd.concat([self._xrestaurants,result], axis=1).sort_values('final_stars', ascending=False)[:num]
        return result
    
    def simple_predict(self):
        result = pd.DataFrame(self._deep_fm_model.predict(self._predict_df), columns=['deepfm_stars'])
        result = pd.concat([self._xrestaurants['business_id'],result], axis=1)
        return result


### Ensemble ##################
class Ensemble():
    def __init__(self, _data_obj):
        self._data_object = _data_obj
        self._xlearn = PredictXLearn(self._data_object)
        self._deepfm = PredictDeepFM(self._data_object)
        
    def predict(self, _user_id, _num = 10):
        self._xlearn.build_libffm(_user_id)
        self._deepfm.build_df2xy(_user_id)        
        
        xlearn = self._xlearn.simple_predict()
        deepfm = self._deepfm.simple_predict()
                                 
        result = self._data_object.restaurant_whole_df
        result = result.merge(xlearn, on='business_id', how='left').merge(deepfm, on='business_id', how='left')
        
        scaler = MinMaxScaler(feature_range=(1,10))
        result['deepfm_stars'] = scaler.fit_transform(result[['deepfm_stars']])
        result['xlearn_stars'] = scaler.fit_transform(result[['xlearn_stars']])
        #result['svd_stars'] = scaler.fit_transform(result[['svd_stars']])
        result['final_stars'] = result['deepfm_stars']*0.5 + result['xlearn_stars']*0.5
        #result['final_stars'] = result.apply(lambda x : result['svd_stars'] if x['user_id'] not in self._data_object.review_df['user_id'].values else result['deepfm_stars']*0.4 + result['xlearn_stars']*0.3 + result['svd_stars']*0.3)
        result = result.sort_values('final_stars', ascending = False)[:_num]
        
        return result        