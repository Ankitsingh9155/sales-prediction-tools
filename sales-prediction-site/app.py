from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import plotly.offline as pyo

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preview', methods=['POST'])
def preview():
    file = request.files['file']
    if file:
        try:
            df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
            df.columns = df.columns.str.strip()  # Remove any spaces in column names
            df.to_csv('uploaded.csv', index=False)
            return render_template('preview.html', tables=[df.head().to_html(classes='data')], titles=df.columns.values)
        except Exception as e:
            return f"File read error: {str(e)}"
    return "File upload error"

@app.route('/predict')
def predict():
    try:
        df = pd.read_csv('uploaded.csv')
        df.columns = df.columns.str.strip()  # Clean column names

        # Check if required columns exist
        if 'Date' not in df.columns or 'Sales' not in df.columns:
            return "Error: The file must contain 'Date' and 'Sales' columns."

        # Convert date column and clean data
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Create numeric index for model
        df['Day'] = range(len(df))

        # Train the model
        model = LinearRegression()
        X = df[['Day']]
        y = df['Sales']
        model.fit(X, y)

        # Predict sales
        df['Predicted'] = model.predict(X)

        # Plot actual vs predicted sales
        trace1 = go.Scatter(x=df['Date'], y=df['Sales'], mode='lines+markers', name='Actual Sales')
        trace2 = go.Scatter(x=df['Date'], y=df['Predicted'], mode='lines', name='Predicted Sales')
        layout = go.Layout(title='Sales Forecast', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        graph_html = pyo.plot(fig, output_type='div')

        return render_template('predict.html', graph=graph_html, tables=[df[['Date', 'Sales', 'Predicted']].to_html(classes='data')])
    except Exception as e:
        return f"Prediction error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
