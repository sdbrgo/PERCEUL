from sklearn.base import BaseEstimator, TransformerMixin

class NumericSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.numeric_cols_ = X.select_dtypes(include=[float, int]).columns
        return self

    def transform(self, X):
        return X[self.numeric_cols_]