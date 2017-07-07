import numpy as np


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        X_array = np.array([np.array(dd) for dd in X_df['spectra']])
        return X_array
