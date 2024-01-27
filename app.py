from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Initialize Flask application
app = Flask(__name__)

# Load your trained model
model = load_model('Elecwa.h5')

# Define MinMaxScaler for normalization
scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the POST request in JSON format
        input_data = request.get_json()

        # Check if 'data' key is present in the JSON
        if 'data' not in input_data:
            return jsonify({'error': 'Missing "data" key in the JSON request'}), 400

        # Extract data from the JSON
        data = input_data['data']

        # Validate that 'data' is a list
        if not isinstance(data, list):
            return jsonify({'error': '"data" should be a list of values'}), 400

        # Convert the list to a NumPy array
        data_array = np.array(data)

        # Convert NumPy array to a standard Python list
        data_list = data_array.astype(float).tolist()

        # Normalize the input data
        normalized_input = scaler.fit_transform(np.array(data_list).reshape(1, -1))

        # Reshape the input data to match the shape expected by the model
        reshaped_input = normalized_input.reshape((normalized_input.shape[0], normalized_input.shape[1], 1))

        # Make predictions
        predicted_value = model.predict(reshaped_input)[0][0]

        # Convert the prediction to a standard Python float
        predicted_value = float(predicted_value)

        # Format the predicted value as a JSON response
        response = {'predicted_value': predicted_value}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)