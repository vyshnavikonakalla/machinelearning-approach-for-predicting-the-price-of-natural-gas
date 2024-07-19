from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

with open('model_dt.pkl', 'rb') as f:
    model = pickle.load(f)

data = pd.read_csv('preprocessed_natural_gas_prices.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    
    date_index = pd.Timestamp(year=year, month=month, day=day)
    
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
    past_data = data[data['Date'] <= date_index].sort_values(by='Date').tail(7) 
    
    price_lag1 = past_data.iloc[-2]['Price'] if len(past_data) > 1 else np.nan
    price_lag7 = past_data.iloc[0]['Price'] if len(past_data) == 7 else np.nan
    price_rolling_mean7 = past_data['Price'].mean()
    
    features = [price_lag1, price_lag7, price_rolling_mean7]
    final_features = [np.array(features)]
    
    prediction = model.predict(final_features)
    
    return render_template('index.html', prediction_text=f'Predicted Natural Gas Price: ${prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
