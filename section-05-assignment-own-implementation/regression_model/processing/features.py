import numpy as np


class ExtractLetterTransformer():
    # Extract fist letter of variable

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    def _get_first_cabin(self, row):

        try:
            return row[:1]
        except ValueError:
            return np.nan

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()
        for var in self.variables:
            X[var] = X[var].apply(self._get_first_cabin)
        return X
