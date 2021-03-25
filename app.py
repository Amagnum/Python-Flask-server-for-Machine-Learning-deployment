# MAIN SERVER

from flask_cors import CORS, cross_origin
from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify, render_template, send_file
from augmentations import augmentations as augs
from baselineModel import baselinePredict
from functions import functions as fnc
from keras.models import load_model, Model
from werkzeug.utils import secure_filename
from PIL import Image
import os
import io
import sys
import numpy as np
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
        'http://127.0.0.1:3000',
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
app.config['CORS_HEADERS'] = 'Content-Type'
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

# TODO Multiclass
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
            image_name = 'predict_0.jpg'
            files[key].save(os.path.join('./temp/predict/', image_name))
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
            'existing_class': 'N' if className == 'add_new' else 'Y',
            'path': image_names,
            'class_id': className,
            'csv_file_path': '/content/drive/MyDrive/Inter_IIT_German_Traffic_Sign/German_Traffic_Sign_Recognition_Dataset/training_set/images_color.csv',  # TODO add CSV PATH
            'save_images_directory': './temp/retrain/'
        }
        status = fnc.upload_class(param)

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
    #augs_list['prob'] = 1.0

    path = ''
    if request.form['input_type'] == 'predict':
        path = './temp/predict/predict_0.jpg'
    elif request.form['input_type'] == 'retrain':
        path = './temp/retrain/retrain_0.jpg'

    image = augs.applyLoadedImage(augs_list, path)

    cv2.imwrite('./display_images/'+request.form['input_type']+'_aug_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

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
    prepPara = {}
    for key in request.form:
        prepPara[key] = json.loads(request.form[key])

    print(list(prepPara.keys()))
    print(list(prepPara.values()))
    images = fnc.preprocessing([img], list(prepPara.keys()), list(prepPara.values()))
    image = images[0]

    cv2.imwrite('./display_images/predict_prep_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
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


retrain_model = None
compiled_model = None
trained_model = None
X_values = []
Y_values = []
n_classes = 1
xtrain = []
xtest = []
ytrain = []
ytest = []
model_name = 'new_model'


@app.route('/retrain_balance', methods=['POST'])
def balance():
    '''
    For Balancing data
    '''
    global augs_list
    global X_values
    global Y_values
    global n_classes

    param = request.form

    print(request.form)

    classKeys = json.loads(param['class_ids'])
    classImgs = json.loads(param['class_numbers'])

    param = {
        'class_ids': classKeys,
        'number_of_images': classImgs,
        'common_val': param['common_val'],
        'input_aug': augs_list,
        'csv_path': '/content/drive/MyDrive/Inter_IIT_German_Traffic_Sign/German_Traffic_Sign_Recognition_Dataset/training_set/images_color.csv'
    }

    X_values, Y_values = fnc.balanced_dataset(param)

    n_classes = len(set(Y_values))
    #cv2.imwrite('./display_images/retrain_aug_image.png', cv2.cvtColor(X_values[0], cv2.COLOR_RGB2BGR))

    data = {
        'status': 'success',
        'nValues': len(X_values),
        'nClasses': n_classes
    }

    return data


@app.route('/retrain_preprocess', methods=['POST'])
def retrain_preprocess():
    '''
    For rendering results on HTML GUI
    '''
    global augs_list
    global X_values

    prepPara = {}
    for key in request.form:
        prepPara[key] = json.loads(request.form[key])

    print(list(prepPara.keys()))
    print(list(prepPara.values()))

    print(X_values)
    print(len(X_values))

    images = fnc.preprocessing(X_values, list(prepPara.keys()), list(prepPara.values()))
    X_values = images
    image = images[0]

    cv2.imwrite('./display_images/retrain_prep_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    img = Image.fromarray(image.astype('uint8'))
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG')


@app.route('/retrain_segregate', methods=['POST'])
def retrain_segregate():
    '''
    For rendering results on HTML GUI
    '''
    global augs_list
    global X_values, Y_values
    global xtrain, xtest, ytrain, ytest

    model_path = '/content/drive/MyDrive/Inter_IIT_German_Traffic_Sign/German_Traffic_Sign_Recognition_Dataset/Sakshee_GTSRB_classification.h5'
    name = request.form['segregation_type']
    test_size = request.form['test_size']

    model = load_model(model_path)
    ypred = fnc.predict_model(model, X_values)
    xtrain, xtest, ytrain, ytest = fnc.split(X_values, Y_values, model_path,
                                             test_size, name, y_pred)

    data = {
        'status': 'success',
        'nTrain': len(xtrain),
        'nTest': len(xtest)
    }
    return data

layerNames = []
layerValues = []

@app.route('/cnn_add_layer', methods=['POST'])
def cnn_add_layer():
    '''
    For rendering results on HTML GUI
    '''
    global layerNames, layerValues

    if request.form['add_mode'] == 'undo' :
      layerNames.pop()
      layerValues.pop()
    elif request.form['add_mode'] == 'reset':
      layerNames = []
      layerValues = []
    else: 
      layerNames.append(request.form['layer_name'])
      layerValues.append(json.loads(request.form['layer_value']))
    
    
    print(layerNames)
    print(layerValues)
    print(json.loads(request.form['layer_value']))

    data = {
        'status': 'success',
    }
    return data



@app.route('/model_settings', methods=['POST'])
def model_settings():
    '''
    For rendering results on HTML GUI
    '''

    global retrain_model
    global n_classes
    global model_name, layerNames, layerValues

    param = {}
    for key in request.form:
        param[key] = json.loads(request.form[key])

    model_name = request.form['model_name']

    if request.form['retrain_type'] == 'Design':
        param = {
            'img_size': 32,
            'channels': request.form['channels'],
            'num_classes': n_classes,
            'list1': layerNames,  # ['Conv2D', 'MaxPool2D', 'Flatten', 'Dense'],
            'list2': layerValues  # [[16, 3, 1, 'same', 'relu'], [2, 1, 'same'], [], [64, 'relu']]
        }
        retrain_model = fnc.design_CNN(param)
    elif request.form['retrain_type'] == 'Pretrained':
        param = {
            'img_size': 32,
            'channels': request.form['channels'],
            'num_classes': n_classes,
            'list1': [request.form['pretrained_model']]
        }
        retrain_model = fnc.pre_trained_softmax(param)
    
    print(retrain_model)

    data = {
        'status': 'success'
    }

    return data


@app.route('/compile', methods=['POST'])
def compile():
    '''
    For rendering results on HTML GUI
    '''

    global retrain_model
    global compiled_model
    global n_classes

    param = {}
    for key in request.form:
        param[key] = json.loads(request.form[key])

    compiled_model = fnc.compile_model(param, retrain_model)
    print(compiled_model)

    data = {
        'status': 'success'
    }
    return data


@app.route('/train_model', methods=['POST'])
def train_model():
    '''
    For rendering results on HTML GUI
    '''

    global compiled_model
    global trained_model
    global model_name
    global xtrain, xtest, ytrain, ytest

    param = {}
    for key in request.form:
        param[key] = json.loads(request.form[key])


    batch_size = request.form['batch_size']
    epochs = request.form['epochs']
    trained_model = fnc.train_model(compiled_model, model_name, batch_size, epochs, xtrain, ytrain, xtest, ytest)
    print(trained_model)


    ypred = fnc.predict_model(trained_model, X_values)
    #visualize_all(xtrain, xtest, ,)

    data = {
        'status': 'success'
    }

    return data

################################################
############# END APIS #########################
################################################

#CORS(app, resources={ r'/*': {'origins': config['ORIGINS']}}, supports_credentials=True)
#CORS(app, resources={ r'/*': {"origins": "http://localhost:port"}}, supports_credentials=True)


CORS(app, supports_credentials=True)

if __name__ == "__main__":
    app.run()
