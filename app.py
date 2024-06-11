import flask
from flask import render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # Example preprocessor
import pickle  # For model loading

# Load pre-trained model
model = pickle.load(open(r"C:\Users\Gaurav\OneDrive\Desktop\codsoft project3\codsoft 3 sales production\eda\linear.pkl", "rb"))

app = flask.Flask(__name__)

@app.route("/")
def index():
    # Provide initial form data (optional)
    form_data = {}  # Empty dictionary to hold form data
    return render_template("index.html", form_data=form_data)

@app.route("/predict_datapoint", methods=["POST"])
def predict_datapoint():
    # Get user input from the form
    TV = request.form.get('TV')
    radio = request.form.get('Radio')
    newspaper = request.form.get('Newspaper')
   # Add more fields as needed

    # Preprocess data (adapt based on your model's requirements)
    data_df = pd.DataFrame({"data1": [TV], "data2": [radio]," data3":[newspaper]})  # Create DataFrame
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_df)

    # Make prediction using the model
    prediction = model.predict(scaled_data)

    # Example analysis using prediction and preprocessed data (replace with your desired analysis)
    predict_datapointe = f"Prediction: {prediction[0]}"  # Assuming single prediction output

    return render_template("results.html", predict_datapoint=predict_datapoint)

if __name__ == "__main__":
    app.run(debug=True)



