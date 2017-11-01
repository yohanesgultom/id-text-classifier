"""
Predict sentence(s) binary class using trained model
@author yohanes.gultom@gmail.com
"""

import pickle
import argparse
import sys
from train import (
  SimpleIndonesianPreprocessor,
  identity
)

MODEL_FILE = 'model.pkl'

parser = argparse.ArgumentParser(description='Classify text based on trained model')
parser.add_argument('sentences', nargs='+', help='one or more sentences each wrapped by double quotes')
parser.add_argument('-m', '--model', default=MODEL_FILE, help='trained model path (default: model.pkl)')
args = parser.parse_args()

with open(args.model, 'rb') as f:
    model = pickle.load(f)

X = args.sentences
y_predict = model.predict(X)

print('Prediction(s):')
for s, label in zip(X, y_predict):
  print('{} ({})'.format(s, label))
