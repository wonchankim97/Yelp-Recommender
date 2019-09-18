"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.
This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import numpy as np
from sklearn.externals import joblib
import re

# Load the models 
# model_dict is the collection of extra tree models 

# This line doesn't work, joblib only loads locally. File is too big to upload to heroku though
# model_dict = joblib.load('https://drive.google.com/open?id=1h20N5Cooti2e5CDkmKY5LOzRuLksyR5e')
# model_dict = joblib.load('./static/models/models_compressed.p')
# word_vectorizer = joblib.load('static/models/word_vectorizer.p')

model_dict = joblib.load('./static/models/log_models.p')
word_vectorizer = joblib.load('static/models/log_word_vectorizer.p')

cl_path = 'static/cleaning/clean_letters.txt'

clean_word_dict = {}
with open(cl_path, 'r', encoding='utf-8') as cl:
    for line in cl:
        line = line.strip('\n')
        typo, correct = line.split(',')
        clean_word_dict[typo] = correct


def raw_chat_to_model_input(raw_input_string):
    # Converts string into cleaned text
    cleaned_text = []
    for text in [raw_input_string]:
        cleaned_text.append(clean_word(text))
    return word_vectorizer.transform(cleaned_text)

def predict_toxicity(raw_input_string):
    ''' Given any input string, predict the toxicity levels'''
    model_input = raw_chat_to_model_input(raw_input_string)
    results = []
    for key,model in model_dict.items():
        results.append(round(model.predict_proba(model_input)[0,1],3))
    return results

def make_prediction(input_chat):
    """
    Given string to classify, returns the input argument and the dictionary of 
    model classifications in a dict so that it may be passed back to the HTML page.
    Input:
    Raw string input
    Function makes sure the features are fed to the model in the same order the
    model expects them.
    Output:
    Returns (x_inputs, probs) where
      x_inputs: a list of feature values in the order they appear in the model
      probs: a list of dictionaries with keys 'name', 'prob'
    """

    if not input_chat:
        input_chat = ' '
    if len(input_chat) > 500:
        input_chat = input_chat[:500]
    pred_probs = predict_toxicity(input_chat)

    probs = [{'name': list(model_dict.keys())[index], 'prob': pred_probs[index]}
             for index in np.argsort(pred_probs)[::-1]]

    return (input_chat, probs)

# This section checks that the prediction code runs properly
# To test, use "python predictor_api.py" in the terminal.

# if __name__='__main__' section only runs
# when running this file; it doesn't run when importing

if __name__ == '__main__':
    from pprint import pprint
    print("Checking to see what empty string predicts")
    print('input string is ')
    chat_in = 'bob'
    pprint(chat_in)

    x_input, probs = make_prediction(chat_in)
    print(f'Input values: {x_input}')
    print('Output probabilities')
    pprint(probs)