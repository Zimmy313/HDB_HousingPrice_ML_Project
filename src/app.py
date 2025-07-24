# app.py
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

#from predictor import predict_price  # to be created
from predictor import predict_price

app = Dash(__name__)
app.title = "HDB Re-sale Price Predictor"

# Define colors
colors = {
    'background': '#FAF7F3',
    'text': '#000000'
}

# town options
town_options = [{'label': town, 'value': town} for town in [
    'BUKIT BATOK',
    'HOUGANG',
    'PUNGGOL',
    'CENTRAL AREA',
    'SERANGOON',
    'BEDOK',
    'JURONG WEST',
    'CLEMENTI',
    'BUKIT PANJANG',
    'ANG MO KIO',
    'BISHAN',
    'YISHUN',
    'TAMPINES',
    'CHOA CHU KANG',
    'QUEENSTOWN',
    'SENGKANG',
    'BUKIT MERAH',
    'WOODLANDS',
    'MARINE PARADE',
    'KALLANG/WHAMPOA',
    'GEYLANG',
    'JURONG EAST',
    'TOA PAYOH',
    'SEMBAWANG',
    'PASIR RIS',
    'BUKIT TIMAH',
    'LIM CHU KANG'
]]

# model options
model_options = [{'label': m, 'value': m} for m in [
 'MODEL A',
 'IMPROVED',
 'NEW GENERATION',
 'SIMPLIFIED',
 'MAISONETTE',
 'PREMIUM APARTMENT',
 'STANDARD',
 'APARTMENT',
 'MODEL A2',
 'DBSS',
 '2-ROOM',
 'MODEL A-MAISONETTE',
 'TYPE S1',
 'ADJOINED FLAT',
 'MULTI GENERATION',
 'TERRACE',
 'TYPE S2',
 'PREMIUM MAISONETTE',
 'PREMIUM APARTMENT LOFT',
 'IMPROVED-MAISONETTE',
 '3GEN'
]]

X_train_1 =pd.read_csv("../data/boost/X_train_boost_part1.csv", index_col = "index")
X_train_2 =pd.read_csv("../data/boost/X_train_boost_part2.csv", index_col = "index")
X_train_3 =pd.read_csv("../data/boost/X_train_boost_part3.csv", index_col = "index")
y_train = pd.read_csv("../data/y_train.csv", index_col = "index")
df = pd.concat([X_train_1,X_train_2,X_train_3])
df = pd.merge(df, y_train, how="inner", on = "index")


def build_trend_layout(df):
    # Ensure month is datetime
    df['month'] = pd.to_datetime(df['month'])

    # Group or aggregate if needed (optional)
    df_grouped = df.groupby('month')['resale_price'].mean().reset_index()

    fig = px.line(df_grouped, x='month', y='resale_price',
                  title='Average Resale Price Over Time')
    
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        title_font=dict(size=22),
        xaxis_title="Year",
        yaxis_title="Average Price"
    )

    return html.Div([
        dcc.Graph(figure=fig)
    ])
    
    

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Predict Price', children=[
            html.Div(style={'backgroundColor': colors['background'], 'padding': '20px'}, children=[

    html.H1("HDB Resale Price Predictor", style={'textAlign': 'center'}),

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


    html.Label("Flat Model"),
    dcc.Dropdown(
        id='input-model',
        options=model_options,
        placeholder="Select model type",
        style={'marginBottom': '20px'}
    ),

    html.Label("Floor Area (sqm)"),
    html.Div([
    dcc.Slider(
        id='input-floor-area',
        min=30,
        max=400,
        step=1,
        value=10,
        marks={i: f'{i}' for i in range(30, 400, 20)},
        tooltip={"placement": "bottom", "always_visible": True},
    )
], style={'marginBottom': '40px'}),

    html.Label("Remaining Lease (years)"),
    html.Div([
    dcc.Slider(
        id='input-lease',
        min=10,
        max=99,
        step=1,
        value=10,
        marks={i: f'{i}' for i in range(10, 100, 10)},
        tooltip={"placement": "bottom", "always_visible": True},
    )
], style={'marginBottom': '40px'}),
    
    html.Label("When do you want to sell your flat?"),
    dcc.Slider(
    id='input-year',
    min=2025,
    max=2125,
    step=0.1,  # Monthly precision
    value=2025,
    marks={i: str(i) for i in range(2025, 2126, 5)},
    tooltip={"placement": "bottom", "always_visible": False},  # Disable built-in tooltip
    updatemode='drag'
), 

    html.Div([
        html.Button("Predict", id='submit-button', n_clicks=0, style={
                        'backgroundColor': '#6c5ce7',
                        'color': 'white',
                        'fontSize': '18px',
                        'border': 'none',
                        'padding': '10px 20px',
                        'borderRadius': '5px',
                        'cursor': 'pointer'
})

    ], style={'textAlign': 'center', 'marginTop': '20px'}),
    
    dcc.Loading(
    id="loading-spinner",
    type="circle",  # types: 'default', 'circle', 'dot', 'cube'
    color="#6c5ce7",
    children=html.Div(
        id='prediction-output',
        style={
            'fontSize': '24px',
            'textAlign': 'center',
            'color': '#5A3E36',
            'marginTop': '30px',
            'padding': '20px',
            'border': '2px solid #E0DCD3',
            'borderRadius': '10px',
            'backgroundColor': '#FFF9F3'
        }
    )
)
])  
        ]),
        dcc.Tab(label='Trend Explorer', children=[
            html.Div(style={'backgroundColor': colors['background'], 'padding': '20px'},
                     id='trend-layout', children=build_trend_layout(df)) 
        ])
    ])
])

# Callback. This means that When the button is clicked, and there are values in area/type/year, call the function.
@callback(
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('input-floor-area', 'value'),
    Input('input-type', 'value'),
    Input('input-year', 'value'),
    Input('input-town', 'value'),
    Input('input-model', 'value'),
    Input('input-lease', 'value')
)
def update_output(n_clicks, area, flat_type, year, town,  model_type, lease):
    
    if n_clicks == 0:
        return ""

    missing = []
    if not area:
        missing.append("Floor Area")
    if not flat_type:
        missing.append("Flat Type")
    if not year:
        missing.append("Year")
    if not town:
        missing.append("Town")
    if not model_type:
        missing.append("Flat Model")
    if not lease:
        missing.append("Remaining Lease")

    if missing:
        return f"Please fill in: {', '.join(missing)}."

    price = predict_price(area, flat_type, year, town, model_type, lease)
    return f"Estimated resale price: ${price:,.2f}"


if __name__ == '__main__':
    app.run_server(debug=True)
