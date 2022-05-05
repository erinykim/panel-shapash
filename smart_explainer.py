import panel as pn
from shapash import SmartExplainer

from loader.data_loader import DataLoader
from transformer.encoder import Encoder
from models import split, model_fit


X_df, y_df, house_dict = DataLoader().call()
encoder, X_df_encoded = Encoder(X_df).call()
X_train, X_test, y_train, y_test = split.Split(X_df_encoded, y_df, train_size=0.7, random_state=1).call()
regressor_model , y_pred = model_fit.ModelFit(X_train, y_train, X_test, n_estimators=200).call()

xpl = SmartExplainer(
    model = regressor_model,
    preprocessing = encoder,
    features_dict = house_dict
)

xpl.compile(x = X_test, y_pred = y_pred)

pn.extension('plotly')

logo = "assets/shapash-resize.png"
material = pn.template.MaterialTemplate(logo=logo, title='Shapash Panel Application', sidebar_width=200)

text = "<h1>Panel Application for Shapash!<h1>"
header = pn.Row(text)

plot_features_pane = pn.pane.Plotly(
    xpl.plot.features_importance(),
    config={'responsive': True}
)

@pn.depends(plot_features_pane.param.click_data)
def update_plot(click_data):
    if click_data:
        plot = pn.pane.Plotly(
            xpl.plot.contribution_plot(click_data['points'][0]['label']),
            config={'responsive': True}
        )
        def local_plot(click_data):
            if click_data:
                return pn.pane.Plotly(
                    xpl.plot.local_plot(index=click_data['points'][0]['customdata'])
                )
        return pn.Column(
            plot,
            pn.depends(click_data=plot.param.click_data)(local_plot)
        )
    else:
        plot = pn.pane.Plotly(
            xpl.plot.contribution_plot(xpl.features_imp.idxmax()),
            config={'responsive': True}
        )
        def local_plot(click_data):
            if click_data:
                return pn.pane.Plotly(
                    xpl.plot.local_plot(index=click_data['points'][0]['customdata'])
                )
        return pn.Column(
            plot,
            pn.depends(click_data=plot.param.click_data)(local_plot)
        )


material.main.append(
    pn.Row(
        pn.Column(
            header
        ),
        pn.Column(
            pn.Tabs(
                ('Smart Explainer', 
                    pn.Column(
                        plot_features_pane,
                        update_plot
                        )
                )
            )
        )
    )
)

material.servable()