from sklearn import linear_model
import scipy.sparse as sps
import numpy as np
import math
import random
from sklearn.model_selection import GridSearchCV
import pickle
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVR

from data_preprocess import get_data


# Data
(train_data, train_label), (test_data, test_label) = get_data()
from IPython import embed

# Random Forest
regr = RandomForestClassifier(n_jobs=-1)

num_estimator = [50,100,200,500]
max_feature = ["auto","sqrt","log2"]
min_samples_leaf = [10,50,100]

tuned_parameters = [{'n_estimators': num_estimator,'max_features':max_feature,'min_samples_leaf':min_samples_leaf}]
n_folds = 10
clf = GridSearchCV(regr, tuned_parameters, cv=n_folds)

# clf = RandomForestClassifier(n_jobs=-1, n_estimators=50, max_features='sqrt', min_samples_leaf=50)

clf.fit(train_data, train_label)
n_folds = 10


# Error Estimation 
print("train score:")
train_scores = cross_val_score(clf, train_data, train_label, cv=n_folds)
print(train_scores.mean())
y_pred_train = clf.predict(train_data)
print(classification_report(train_label, y_pred_train, target_names=['False', 'True']))
# FIXME: not sure....
# y_pred_train = clf.predict(train_data)
# print(mean_squared_error(train_label,y_pred_train))
# results = sm.OLS(y_pred_train,sm.add_constant(train_data)).fit()
# print(results.summary())

print("test score:")
test_scores = cross_val_score(clf, test_data, test_label, cv=n_folds)
print(test_scores.mean())
y_pred_test = clf.predict(test_data)
print(classification_report(test_label, y_pred_test, target_names=['False', 'True']))
# y_pred_test = clf.predict(test_data)
# print(mean_squared_error(test_label,y_pred_test))
# results = sm.OLS(y_pred_test,sm.add_constant(test_data)).fit()
# print(results.summary())


#########  Saving model
rf_pkl_filename = 'rf_20190326.pkl'
rf_model_pkl = open(rf_pkl_filename, 'wb')
pickle.dump(clf,rf_model_pkl)
rf_model_pkl.close()

########## Saving prediction test
rf_pkl_test_filename = 'rf_20190326_test_pred.pkl'
rf_test_pkl = open(rf_pkl_test_filename, 'wb')
pickle.dump(y_pred_test,rf_test_pkl)
rf_test_pkl.close()

######### Saving Real label test
rf_pkl_test_filename = 'rf_20190326_test_real_label.pkl'
rf_test_pkl = open(rf_pkl_test_filename, 'wb')
pickle.dump(test_label,rf_test_pkl)
rf_test_pkl.close()


########## Saving prediction train
rf_pkl_train_filename = 'rf_20190326_train_pred.pkl'
rf_train_pkl = open(rf_pkl_train_filename, 'wb')
pickle.dump(y_pred_train,rf_train_pkl)
rf_train_pkl.close()

######### Saving Real label train
rf_pkl_train_filename = 'rf_20190326_train_real_label.pkl'
rf_train_pkl = open(rf_pkl_train_filename, 'wb')
pickle.dump(train_label,rf_train_pkl)
rf_train_pkl.close()
