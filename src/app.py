# app.py
from dash import Dash, html, dcc, callback, Output, Input
#from model.predictor import predict_price  # to be created
import pandas as pd

app = Dash(__name__)
app.title = "HDB Price Predictor"

# Define colors
colors = {
    'background': '#FAF7F3',
    'text': '#000000'
}

# town options
town_options = [{'label': town, 'value': town} for town in [
    'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT MERAH', 'CHOA CHU KANG'
]]

# model options
model_options = [{'label': m, 'value': m} for m in [
    'Improved', 'New Generation', 'Model A', 'Simplified'
]]

app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '20px'}, children=[

    html.H1("HDB Price Predictor", style={'textAlign': 'center'}),

    html.Label("Flat Type"),
    dcc.Dropdown(
        id='input-type',
        options=[
            {'label': '1 ROOM', 'value': '1 ROOM'},
            {'label': '2 ROOM', 'value': '2 ROOM'},
            {'label': '3 ROOM', 'value': '3 ROOM'},
            {'label': '4 ROOM', 'value': '4 ROOM'},
            {'label': '5 ROOM', 'value': '5 ROOM'},
            {'label': 'EXECUTIVE', 'value': 'EXECUTIVE'},
            {'label': 'MULTI-GENERATION', 'value': 'MULTI-GENERATION'},
        ],
        placeholder="Select flat type",
        style={'marginBottom': '20px'}
    ),

    html.Label("Town"),
    dcc.Dropdown(
        id='input-town',
        options=town_options,
        placeholder="Select town",
        style={'marginBottom': '20px'}
    ),

    html.Label("Block"),
    dcc.Input(
        id='input-block',
        type='text',
        placeholder="Enter block number, e.g. 123",
        style={'width': '100%', 'marginBottom': '20px'}
    ),

    html.Label("Flat Model"),
    dcc.Dropdown(
        id='input-model',
        options=model_options,
        placeholder="Select model type",
        style={'marginBottom': '20px'}
    ),

    html.Label("Floor Area (sqm)"),
    dcc.Slider(
        id='input-floor-area',
        min=30,
        max=150,
        step=1,
        value=90,
        marks={i: f'{i}' for i in range(30, 151, 20)},
        tooltip={"placement": "bottom", "always_visible": True},
        style={'marginBottom': '40px'}
    ),

    html.Label("Remaining Lease (years)"),
    dcc.Slider(
        id='input-lease',
        min=1,
        max=99,
        step=1,
        value=70,
        marks={i: f'{i}' for i in range(1, 100, 10)},
        tooltip={"placement": "bottom", "always_visible": True},
        style={'marginBottom': '40px'}
    ),

    html.Label("Year"),
    dcc.Slider(
        id='input-year',
        min=1990,
        max=2025,
        step=1,
        value=2025,
        marks={i: str(i) for i in range(1990, 2026, 5)},
        tooltip={"placement": "bottom", "always_visible": True},
        style={'marginBottom': '40px'}
    ),

    html.Button("Predict", id='submit-button', n_clicks=0),
    html.Br(), html.Br(),

    html.Div(id='prediction-output', style={'fontSize': '24px', 'textAlign': 'center', 'color': '#5A3E36'})
])

# Callback
@callback(#This means that When the button is clicked, and there are values in area/type/year, call the function.”
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('input-area', 'value'),
    Input('input-type', 'value'),
    Input('input-year', 'value')
)
def update_output(n_clicks, area, flat_type, year):
    if n_clicks > 0 and area and flat_type and year:
        price = predict_price(area, flat_type, year)  # To be implemented
        return f"Estimated resale price: ${price:,.2f}"
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
