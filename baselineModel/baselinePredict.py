import cv2 as cv
import numpy
from keras.models import load_model
from augmentations import augmentations as augs
import matplotlib.pyplot as plt

def load_image(path, model):
    '''
    function: loads image uploaded by user
    :param params: params['path'] = C:/users/desktop/uploads/image.jpg
     params['size'] = size of image
    :return: image
    '''
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  
    if 'yuvnish' in model:
      size = 64
    else:
      size = 32
    #Yuvnish's model-> params['size']=64
    #Sakshee's model-> params['size']=32
    #size = 32
    image = cv.resize(image, (size, size))
    return image

def load_my_model(path):
    """
    function: loads pre-trained neural network
    :param params: params['path'] = 'C:/users/desktop/uploads/model_v1.h5'
    :return: model
    """
    model = load_model(path)
    return model

def predict(params, aug_list):
    '''
    function: gives prediction on single image
    :param params: params['path_image'] = 'C:/users/desktop/uploads/image.jpg'
                 params['path_model'] = 'C:/users/desktop/uploads/model_v1.h5'
    :return: prediction_list (a list), class_id (an integer)
    '''

    image = load_image(params['path_image'],params['path_model'])
    image = augs.applyAugmentation(image, aug_list)  # augmentation function
    cv.imwrite('./display_images/predict_aug_image.png', image)
    # image = preprocess(image)  # preprocess function
    cv.imwrite('./display_images/predict_prep_image.png', image)
    model = load_my_model(params['path_model'])
    image = numpy.array(image)
    image = numpy.expand_dims(image, axis=0)
    prediction = model.predict(image)
    print(prediction)
    prediction = prediction.tolist()
    m = max(prediction[0])
    class_id = prediction[0].index(m)
    prediction_list = prediction[0]
    return prediction_list, class_id