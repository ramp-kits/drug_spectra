import os
import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

problem_title =\
    'Drug classification and concentration estimation from Raman spectra'
_prediction_label_names = ['A', 'B', 'Q', 'R']
_target_column_name_clf = 'molecule'
_target_column_name_reg = 'concentration'
Predictions_1 = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# We make a 2D but single-column y_pred (instead of a classical 1D y_pred)
# to make handling the combined 2D y_pred array easier
Predictions_2 = rw.prediction_types.make_regression(
    label_names=[_target_column_name_reg])
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_combined([Predictions_1, Predictions_2])
# An object implementing the workflow
workflow = rw.workflows.DrugSpectra()

score_type_1 = rw.score_types.ClassificationError(name='error', precision=3)
score_type_2 = rw.score_types.MARE(name='mare', precision=3)
score_types = [
    rw.score_types.Combined(
        name='combined', score_types=[score_type_1, score_type_2],
        weights=[2. / 3, 1. / 3], precision=3),
    rw.score_types.MakeCombined(score_type_1, 0),
    rw.score_types.MakeCombined(score_type_2, 1)
]


def _read_data(path, f_name):
    df = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = df[[_target_column_name_clf, _target_column_name_reg]].values
    X_df = df.drop([_target_column_name_clf, _target_column_name_reg], axis=1)
    spectra = X_df['spectra'].values
    spectra = np.array([np.array(
        dd[1:-1].split(',')).astype(float) for dd in spectra])
    X_df['spectra'] = spectra.tolist()
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv.bz2'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv.bz2'
    return _read_data(path, f_name)


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X)
