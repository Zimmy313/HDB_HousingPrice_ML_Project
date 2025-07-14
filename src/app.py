from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

# initialiser
app = Dash(__name__) 


# Define the layout: what appears on screen
app.layout = html.Div([
    html.H1("HDB Price Predictor"),
    html.Label("Floor Area(sqm):"),
    dcc.Input(id = 'input-area', type='number', placeholder=' e.g. 85'),
    
    html.Br(), html.Br(),
    
    html.Label("Flat Type:"),
    dcc.Dropdown(
        id = 'input-type',
        options=[
            {'label': '1 ROOM', 'value': '1'},
            {'label': '2 ROOM', 'value': '2'},
            {'label': '3 ROOM', 'value': '3'},
            {'label': '4 ROOM', 'value': '4'},
            {'label': '5 ROOM', 'value': '5'},
            {'label': 'Multi-Generation', 'value': '6'},
            {'label': 'EXECLUSIVE', 'value': '7'},
        ],
        placeholder= "Select flat type"
    ),
    html.Br(), html.Br(),
    
    html.Label("Year:"),
    dcc.Input(id='input-year', type = 'number', placeholder=' e.g. 2030'),
    
    
    html.Br(), 
    
    html.Button("Predict", id='submit-button', n_clicks = 0),
    html.Br(), html.Br(),
    
    html.Div(id='prediction-output')
])

if __name__ == '__main__':
    app.run_server(debug=True)
