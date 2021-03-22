from flask_ngrok import run_with_ngrok
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import pickle
from augmentations import augmentations as augs
from PIL import Image
import os , io , sys
import base64
from flask_cors import CORS

# Cors
config = {
  'ORIGINS': [
    'http://localhost:8080',  # React
    'http://127.0.0.1:8080',  # React
    'http://localhost:3000'
  ],

  'SECRET_KEY': '...'
}

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


@app.route('/augment',methods=['POST'])
def augment():
    '''
    For Augmenting the image files
    '''
    print(request.form)
    image = augs.applyAugmentation(request.form)
    img = Image.fromarray(image.astype('uint8'))
    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG')
    #return render_template('index.html', prediction_text='Employee Salary should be  ${}'.format(request.form))

CORS(app, resources={ r'/*': {'origins': config['ORIGINS']}}, supports_credentials=True)

if __name__ == "__main__":
    app.run()