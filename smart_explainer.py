from shapash import SmartExplainer

from loader.data_loader import DataLoader
from transformer.encoder import Encoder
from models import split, model_fit

X_df, y_df, house_dict = DataLoader().call()
encoder, X_df_encoded = Encoder(X_df).call()
X_train, X_test, y_train, y_test = split.Split(X_df_encoded, y_df, train_size=0.7, random_state=1)
regressor_model , y_pred = model_fit.ModelFit(X_train, y_train, X_test, n_estimators=200)

xpl = SmartExplainer(
    model = regressor_model,
    preprocessing = encoder,
    features_dict = house_dict
)

xpl.compile(x = X_test, y_pred = y_pred)

