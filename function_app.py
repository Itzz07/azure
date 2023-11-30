import azure.functions as func
import logging
import os
import pickle
from prophet import Prophet
import pandas as pd
import json
from datetime import datetime, timedelta

# Add this at the beginning of your script
logging.basicConfig(level=logging.DEBUG)

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# # make prediction
# @app.route(route="predict")
# def predict(req: func.HttpRequest) -> func.HttpResponse:
    
#     # Load the input data from 'unzaweather.csv' file
#     data = pd.read_csv('unzaweather1.csv')

#     # Rename and reformat the columns
#     data.rename(columns={'MaxTemp': 'Temperature', 'Rainfall': 'Rainfall', 'Humidity': 'Humidity'}, inplace=True)
#     data['ds'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

#     # Drop rows with NaN values in the 'ds' column
#     data = data.dropna(subset=['ds'])

#     # Create a dictionary to store models for each parameter
#     models = {}

#     # Train separate Prophet models for each parameter
#     for parameter in ['Temperature', 'Rainfall', 'Humidity']:
#         parameter_data = data[['ds', parameter]].rename(columns={parameter: 'y'})

#         model = Prophet()
#         model.fit(parameter_data)
#         models[parameter] = model

#         # Save the trained model using pickle
#         with open(f'{parameter}_model.pkl', 'wb') as file:
#             pickle.dump(model, file)
    
#     # Make predictions using the loaded models for specific future dates
#     predictions = make_predictions(models)

#     # Convert Timestamp objects to string before serializing to JSON
#     for parameter, forecast in predictions.items():
#         for entry in forecast:
#             entry['ds'] = str(entry['ds'])

#     return func.HttpResponse(json.dumps(predictions), mimetype="application/json")

# # make predictions
# def make_predictions(models):
#     # Generate future dates
#     future_dates = [datetime.now() + timedelta(days=i) for i in [1, 7, 14]]

#     # Create dataframes for predictions
#     future_data = pd.DataFrame({'ds': future_dates})

#     # Make forecasts for each parameter
#     forecasts = {}
#     for parameter, model_path in models.items():
#         # Load the trained model from the pickle file
#         with open(f'{parameter}_model.pkl', 'rb') as file:
#             loaded_model = pickle.load(file)

#         forecast = loaded_model.predict(future_data)
#         # Convert the forecast DataFrame to a dictionary
#         forecasts[parameter] = forecast.to_dict(orient='records')

#     return forecasts

# uploading the data parameters
@app.route(route="upload")
def upload(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        logging.debug(f"Received request: {req_body}")
    except ValueError as ve:
        logging.error(f'Invalid JSON data in the request: {ve}')
        return func.HttpResponse('Invalid JSON data in the request', status_code=400)
    
    # Validate if the required fields are present in the JSON data
    required_fields = ['Location', 'Humidity', 'MaxTemp', 'Rainfall']
    for field in required_fields:
        if field not in req_body:
            return func.HttpResponse(f'Missing "{field}" field in JSON data', status_code=400)

    try:
        # Reorder and validate the data
        ordered_data = {
            'Date': datetime.now().strftime('%d/%m/%Y'),  # Add the current date
            'Time': datetime.now().strftime('%H:%M:%S'),  # Add the current time
            'Location': req_body['Location'],
            'Humidity': float(req_body['Humidity']),
            'MaxTemp': float(req_body['MaxTemp']),
            'Rainfall': float(req_body['Rainfall'])
        }
    except KeyError as ke:
        return func.HttpResponse(f'Missing key in JSON data: {ke}', status_code=400)
    except ValueError as ve:
        return func.HttpResponse(f'Invalid value in JSON data: {ve}', status_code=400)

    # Append the new data to the existing CSV file
    data_to_append = pd.DataFrame([ordered_data])
    
    # Append to the CSV file
    data_to_append.to_csv('unzaweather1.csv', mode='a', header=False, index=False)

    return func.HttpResponse("Data uploaded successfully on azure", status_code=200)

    # return func.HttpResponse('Data saved in the specified order', status_code=200)

# validating date and time
def validate_date(date_string):
    # Validate and convert date to the desired format
    try:
        date = datetime.strptime(date_string, '%d/%m/%Y')
        return date.strftime('%d/%m/%Y')
    except ValueError:
        raise ValueError

def validate_time(time_string):
    # Validate and convert time to the desired format
    try:
        time = datetime.strptime(time_string, '%H:%M:%S').time()
        return time.strftime('%H:%M:%S')
    except ValueError:
        raise ValueError

@app.route(route="predict")
def predict(req: func.HttpRequest) -> func.HttpResponse:
    
    # Load the input data from 'unzaweather.csv' file
    data = pd.read_csv('unzaweather1.csv')

    # Rename and reformat the columns
    data.rename(columns={'MaxTemp': 'Temperature', 'Rainfall': 'Rainfall', 'Humidity': 'Humidity'}, inplace=True)
    data['ds'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

    # Drop rows with NaN values in the 'ds' column
    data = data.dropna(subset=['ds'])

    # Create a dictionary to store models for each parameter
    models = {}

    # Train separate Prophet models for each parameter
    for parameter in ['Temperature', 'Rainfall', 'Humidity']:
        parameter_data = data[['ds', parameter]].rename(columns={parameter: 'y'})

        model = Prophet()
        model.fit(parameter_data)
        models[parameter] = model
    
    # Make predictions using the loaded models for specific future dates
    predictions = make_predictions(models)

    # Convert Timestamp objects to string before serializing to JSON
    for parameter, forecast in predictions.items():
        for entry in forecast:
            entry['ds'] = str(entry['ds'])

    return func.HttpResponse(json.dumps(predictions), mimetype="application/json")

# making predictions
def make_predictions(models):
    # Generate future dates
    future_dates = [datetime.now() + timedelta(days=i) for i in [1, 7, 14]]

    # Create dataframes for predictions
    future_data = pd.DataFrame({'ds': future_dates})

    # Make forecasts for each parameter
    forecasts = {}
    for parameter, model in models.items():
        forecast = model.predict(future_data)
        # Convert the forecast DataFrame to a dictionary
        forecasts[parameter] = forecast.to_dict(orient='records')

    return forecasts
