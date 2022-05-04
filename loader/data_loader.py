from shapash.data.data_loader import data_loading


class DataLoader():
    """
    Class to load data from shapash library.
    data_type includes the data available (ie. house_prices)
    """
    def __init__(self):
        pass

    def call(self):
        house_df, house_dict = self._get_data()
        X_df, y_df = self._transform_data(house_df)
        return X_df, y_df, house_dict

    def _get_data(self):
        house_df, house_dict = data_loading('house_prices')
        return house_df, house_dict

    def _transform_data(self, data):
        y_df = data['SalePrice'].to_frame()
        X_df = data[data.columns.difference(['SalePrice'])]
        return X_df, y_df