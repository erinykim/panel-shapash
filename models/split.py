from sklearn.model_selection import train_test_split


class Split():
    
    def __init__(self, X_df, y_df, train_size, random_state):
        self.X_df = X_df
        self.y_df = y_df
        self.train_size = train_size
        self.random_state = random_state

    def call(self):
        return train_test_split(
            self.X_df, 
            self.y_df, 
            train_size = self.train_size, 
            random_state = self.random_state
        )
