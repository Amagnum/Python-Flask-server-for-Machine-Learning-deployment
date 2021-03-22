import cv2 as cv
import numpy
from keras.models import load_model

def load_image(dict):
    '''
    function: loads image uploaded by user
    :param dict: dict['path'] = C:/users/desktop/uploads/image.jpg
     dict['size'] = size of image
    :return: image
    '''
    
    image = cv.imread(dict['path'])
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    #Yuvnish's model-> dict['size']=64
    #Sakshee's model-> dict['size']=32
    image = cv.resize(image, (dict['size_x'], dict['size_y']))
    return image

def load_model(dict):
    """
    function: loads pre-trained neural network
    :param dict: dict['path'] = 'C:/users/desktop/uploads/model_v1.h5'
    :return: model
    """
    model = load_model(dict['path'])
    return model

def predict(dict):
    '''
    function: gives prediction on single image
    :param dict: dict['path_image'] = 'C:/users/desktop/uploads/image.jpg'
                 dict['path_model'] = 'C:/users/desktop/uploads/model_v1.h5'
    :return: prediction_list (a list), class_id (an integer)
    '''
    
    image = load_image(dict['path_image'])
    image = aug(image)  # augmentation function
    # image = preprocess(image)  # preprocess function
    model = load_model(dict['path_model'])
    image = numpy.array(image)
    image = numpy.expand_dims(image, axis=0)
    prediction = model.evaluate(image)
    prediction = prediction.tolist()
    m = max(prediction[0])
    class_id = prediction[0].index(m)
    prediction_list = prediction[0]
    return prediction_list, class_id