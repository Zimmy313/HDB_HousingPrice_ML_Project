# predictor
import pandas as pd
from catboost import CatBoostRegressor, Pool

from utils import *

model = CatBoostRegressor()
model.load_model("../model/catboost.cbm")

# Define input columns in same order used during training
FEATURE_COLS = [
     'town','flat_type','floor_area_sqm', 'flat_model', 'remaining_lease', 'year'
]

def predict_price(area, flat_type, year, town, flat_model, lease):
    
    # Create a DataFrame with one row
    data = pd.DataFrame([{
        
        'town': town,
        'flat_type': flat_type,
        'floor_area_sqm': area,
        'flat_model': flat_model,
        'remaining_lease': lease,
        'year': int(year)
        
    }])
    
    # Data processing
    data = clean_flat_model(data)
    
    categorical_cols = ['flat_type','town','flat_model']
    for col in categorical_cols:
        data[col] = data[col].astype('category')
        
    prediction_pool = Pool(
    data=data,
    cat_features=['flat_type', 'town', 'flat_model']
)
    
    # Predict
    price = model.predict(prediction_pool)
    return price[0]
