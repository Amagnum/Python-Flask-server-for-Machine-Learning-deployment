#!pip install flask-ngrok
#!pip install flask_cors

#from flask_cors import CORS
from baselineModel import baselinePredict
from flask_ngrok import run_with_ngrok
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import pickle
from augmentations import augmentations as augs
from PIL import Image
from werkzeug.utils import secure_filename
import os , io , sys
import base64
import json


# list to keep track of all the augmentations
augs_list = []

# Cors
config = {
  'ORIGINS': [
    'http://localhost:8080',  # React
    'http://127.0.0.1:8080',  # React
    'http://localhost:3000',
    'http://127.0.0.1:5000'
  ],

  'SECRET_KEY': '...'
}


app = Flask(__name__)

app.config.from_mapping(
        BASE_URL="http://localhost:3000",
)

run_with_ngrok(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    global augs_list
    prediction_list, class_id = baselinePredict.predict(request.form, augs_list)
    data = {
        "prediction_list" : prediction_list,
        "class_id": class_id
    }
    return data

##############
############# Uploads #########################
############

UPLOAD_FOLDER = './Temp'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
'''
@app.route('/upload', methods=['POST'])
def fileUpload():
    target=os.path.join(UPLOAD_FOLDER,'test_docs')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    filename = secure_filename(file.filename)
    destination="/".join([target, filename])
    file.save(destination)
    session['uploadFilePath']=destination
    response= file
    return response
'''
##############
############# Augment #########################
############

@app.route('/augment',methods=['POST'])
def augment():
    '''
    For Augmenting the image files
    '''
    print(request.form)
    global augs_list
    print(augs_list)
    if request.form['aug_mode'] == 'undo' :
        augs_list.pop()
    elif request.form['aug_mode'] == 'reset':
        augs_list = []
    else:
        augs_list.append(request.form)

    print(augs_list)
    image = augs.applyAugmentation(augs_list)
    img = Image.fromarray(image.astype('uint8'))
    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG')
    #return render_template('index.html', prediction_text='Employee Salary should be  ${}'.format(request.form))

#CORS(app, resources={ r'/*': {'origins': config['ORIGINS']}}, supports_credentials=True)

if __name__ == "__main__":
    app.run()