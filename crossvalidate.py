"""
Predict sentence(s) binary class using trained model
@author yohanes.gultom@gmail.com
"""

import pickle
import argparse
import csv
import sys
import numpy as np
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate
from train import (
  SimpleIndonesianPreprocessor,
  cross_validate_result_tabular_report,
  TfidfEmbeddingVectorizer,
  identity
)

RAW_DATASET_FILE = 'dataset_labeled.csv'
MODEL_FILE = 'model.pkl'
CV = 8

parser = argparse.ArgumentParser(description='Classify text based on trained model')
parser.add_argument('-d', '--dataset', default=RAW_DATASET_FILE, help='labeled TSV dataset (default: {})'.format(RAW_DATASET_FILE))
parser.add_argument('-k', '--kfold', default=CV, help='default k for k-fold cross-validation (default: {})'.format(CV))
parser.add_argument('-m', '--model', default=MODEL_FILE, help='trained model path (default: {})'.format(MODEL_FILE))
args = parser.parse_args()

with open(args.model, 'rb') as f:
    model = pickle.load(f)
    model.set_params(preprocessor__verbose=False)

X = []
y = []
with open(args.dataset, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        X.append(row[0])
        y.append(int(row[1]))

# cross validation    
scorer = {
    'pos_recall': make_scorer(recall_score, pos_label=1),
    'neg_recall': make_scorer(recall_score, pos_label=0),
    'pos_precision': make_scorer(precision_score, pos_label=1),
    'neg_precision': make_scorer(precision_score, pos_label=0),
}
cv_results = cross_validate(model, X, y=y, scoring=scorer, cv=args.kfold)

# calculate f1
cv_results['test_pos_f1'] = np.array([ 2 * cv_results['test_pos_recall'][i] * cv_results['test_pos_precision'][i] / (cv_results['test_pos_recall'][i] + cv_results['test_pos_precision'][i]) for i in range(len(cv_results['test_pos_recall'])) ])

cv_results['test_neg_f1'] = np.array([ 2 * cv_results['test_neg_recall'][i] * cv_results['test_neg_precision'][i] / (cv_results['test_neg_recall'][i] + cv_results['test_neg_precision'][i]) for i in range(len(cv_results['test_neg_recall'])) ])

cross_validate_result_tabular_report(cv_results, keys=['test_pos_f1', 'test_pos_precision', 'test_pos_recall', 'test_neg_f1', 'test_neg_precision', 'test_neg_recall'])

