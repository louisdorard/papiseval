import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class simple_model(BaseEstimator, ClassifierMixin):
    """Predicts the majority class of its training data."""
    def __init__(self):
        pass
    def fit(self, X, y):
        self.classes_, indices = np.unique(["foo", "bar", "foo"], return_inverse=True)
        self.majority_ = np.argmax(np.bincount(indices))
        return self
    def predict(self, X):
        return np.repeat(self.classes_[self.majority_], len(X))