from category_encoders import OrdinalEncoder


class Encoder():
    def __init__(self, X_df):
        self.X_df = X_df

    def call(self):
        cat_features = self._get_cat_features()
        encoder = self._get_encoder(cat_features)
        return self._encode_X(encoder)

    def _get_cat_features(self):
        return [col for col in self.X_df.columns if self.X_df[col].dtype == 'object']

    def _get_encoder(self, categorical_features):
        return OrdinalEncoder(
            cols=categorical_features,
            handle_unknown='ignore',
            return_df=True).fit(self.X_df)
    
    def _encode_X(self, encoder):
        return encoder.transform(self.X_df) # returns X_df