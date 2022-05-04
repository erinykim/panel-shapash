import pandas as pd
from lightgbm import LGBMRegressor


class ModelFit():
    
    def __init__(self, X_train, y_train, X_test, n_estimators):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.n_estimators = n_estimators

    def call(self):
        regressor = self._fit_regressor()
        return self._predict(regressor)


    def _fit_regressor(self):
        return LGBMRegressor(n_estimators=self.n_estimators).fit(self.X_train, self.y_train)

    def _predict(self, regressor):
        return pd.DataFrame(
            regressor.predict(self.X_test),
            columns=['pred'],
            index=self.X_test.index
            ) # returns y_pred