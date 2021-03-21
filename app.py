import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import pickle
from augmentations import augmentations as augs
from PIL import Image
import os , io , sys
import base64

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

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
    print(request.form['aug_name'])
    print(request.form['blur_limit'])
    print(request.form['prob'])
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

if __name__ == "__main__":
    app.run(debug=True)