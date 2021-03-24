# MAIN SERVER

from flask_cors import CORS, cross_origin
from flask_ngrok import run_with_ngrok
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import pickle
from augmentations import augmentations as augs
from baselineModel import baselinePredict
from functions import functions as fnc
from PIL import Image
from werkzeug.utils import secure_filename
import os
import io
import sys
import base64
import cv2
import json

##############
############# Variables #########################
############

# list to keep track of all the augmentations
augs_list = []
pre_processing = {}

# Cors
config = {
    'ORIGINS': [
        'http://localhost:8080',  # React
        'http://127.0.0.1:8080',  # React
        'http://localhost:3000',
        'http://127.0.0.1:5000',
        'https://xenodochial-poincare-85c4e7.netlify.app',
        'https://quirky-snyder-121fad.netlify.app/',
        "*"
    ],

    'SECRET_KEY': '...'
}


UPLOAD_FOLDER = './temp'
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

@app.route('/uploadimages', methods=["POST"])
def uploadimages():
    files = request.files
    print(request.files)
    # print(files)
    print(request.form)
    # print(files)
    status = 'false'
    if request.form['input_type'] == 'predict':
        for key in files:
            print(key)
            print(files[key])
            files[key].save(os.path.join('./temp/predict/predict_'+key, 'jpg'))
        status = 'success'
    elif request.form['input_type'] == 'retrain':
        className = request.form['class_name']
        image_names = []
        for key in files:
            print(key)
            print(files[key])
            image_name = 'retrain_'+key+'.jpg'
            image_names.append(image_name)
            files[key].save(os.path.join('./temp/retrain/', image_name))
        # Add data to CSV
        param = {
            'existing_class': 'Y',
            'path': image_names,
            'class_id': className,
            'csv_file_path': '',  # TODO add CSV PATH
            'saved_images_directory': './temp/retrain/'
        }
        #status = fnc.upload_class(param)

    # print(files)
    print(request.form)
    # print(files)
    # print(jsonData)

    return jsonify({"status": status})


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
        id = request.args['id']
    else:
        return "Error: No id field provided. Please specify an id."

    image_path = './display_images/'+str(id)
    if os.path.isfile(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread('./display_images/not_found.png', cv2.IMREAD_COLOR)
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

@app.route('/augment', methods=['POST'])
def augment():
    '''
    For Augmenting the image files
    '''
    print(request.form)
    global augs_list
    # print(augs_list)
    if request.form['aug_mode'] == 'undo':
        augs_list.pop()
    elif request.form['aug_mode'] == 'reset':
        augs_list = []
    else:
        augs_list.append(request.form)

    print(augs_list)
    augs_list['prob'] = 1.0

    path = ''
    
    if request.form['input_type'] == 'predict':
        path = './temp/predict/predict_0.jpg'
    elif request.form['input_type'] == 'retrain':
        path = './temp/retrain/retrain_0.jpg'

    image = augs.applyLoadedImage(augs_list, path)

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


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    global augs_list
    # Sakshee_GTSRB_classification.h5
    # yuvnish_tsr_model_v5.h5
    param = {
        "path_model": "/content/drive/MyDrive/Inter_IIT_German_Traffic_Sign/German_Traffic_Sign_Recognition_Dataset/"+request.form['path_model'],
        "path_image": './display_images/predict_prep_image.png'
    }
    prediction_list, class_id = baselinePredict.predict(param, [])
    data = {
        "prediction_list": prediction_list,
        "class_id": class_id
    }

    return data


@app.route('/predict_preprocess', methods=['POST'])
def predict_preprocess():
    '''
    For rendering results on HTML GUI
    '''
    global augs_list

    param = {}
    param['path'] = './display_images/predict_aug_image.png'
    img = fnc.load_image(param)
    prepPara = request.form

    print(list(prepPara.keys()))
    print(list(prepPara.values()))
    images = fnc.preprocessing([img], list(prepPara.keys()), list(prepPara.values()))
    image = images[0]

    cv.imwrite('./display_images/predict_prep_image.png', image)
    # plt.imshow(images[0])

    img = Image.fromarray(image.astype('uint8'))
    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG')

##############
############# Re-Train #########################
############


@app.route('/retrain', methods=['POST'])
def retrain():
    '''
    For rendering results on HTML GUI
    '''
    global augs_list
    prediction_list, class_id = baselinePredict.predict(request.form, augs_list)
    data = {
        "prediction_list": prediction_list,
        "class_id": class_id
    }

    return data

################################################
############# END APIS #########################
################################################


CORS(app, resources={r'/*': {'origins': config['ORIGINS']}}, supports_credentials=True)

if __name__ == "__main__":
    app.run()
