# MAIN SERVER

from flask_cors import CORS, cross_origin
from flask_ngrok import run_with_ngrok
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import pickle
from augmentations import augmentations as augs
from baselineModel import baselinePredict
from PIL import Image
from werkzeug.utils import secure_filename
import os , io , sys
import base64
import cv2
import json

##############
############# Variables #########################
############

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


UPLOAD_FOLDER = './Temp'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.from_mapping(
        BASE_URL="http://localhost:3000",
)

run_with_ngrok(app)


############################################
############# APIs #########################
############################################

@app.route('/')
def home():
    return render_template('index.html')


##############
############# Uploads #########################
############

@app.route('/uploadimages', methods=['GET','POST'])
def fileUpload():
    target=os.path.join(UPLOAD_FOLDER,'test_docs')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    filename = secure_filename(file.filename)
    destination="/".join([target, filename])
    file.save(destination)
    session['uploadFilePath']=destination
    response= "this is a string"
    return response

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
############# Display Images #########################
############

@app.route('/displayImages')
def displayImage():
    '''
    For serving the display images (used in display, outputs, augmentations, etc)
    http://my_URL.com/displayImages?id=image_name.png
    '''
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    image = cv2.imread('./display_images/'+str(id), cv2.IMREAD_COLOR)
    img = Image.fromarray(image.astype('uint8'))
    # create file-object in memory
    file_object = io.BytesIO()
    # write PNG in file-object
    img.save(file_object, 'PNG')
    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG')


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
    #print(augs_list)
    if request.form['aug_mode'] == 'undo' :
        augs_list.pop()
    elif request.form['aug_mode'] == 'reset':
        augs_list = []
    else:
        augs_list.append(request.form)

    print(augs_list)
    image = augs.applyLoadedImage(augs_list)
    img = Image.fromarray(image.astype('uint8'))
    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG')
    
##############
############# Predict #########################
############

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
############# Re-Train #########################
############

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

################################################
############# END APIS #########################
################################################

CORS(app, resources={ r'/*': {'origins': config['ORIGINS']}}, supports_credentials=True)

if __name__ == "__main__":
    app.run()