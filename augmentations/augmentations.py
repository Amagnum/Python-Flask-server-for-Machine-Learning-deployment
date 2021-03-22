# Commented out IPython magic to ensure Python compatibility.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import augmentations.Automold as am
import augmentations.Helpers as hp
import matplotlib.image as mpimg
import albumentations as A


#path = '../temp/images/*.jpg'
#images = hp.load_images(path)
#image = images[0]
path = './temp/images/img2.jpg'
image = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.savefig("foo.jpg")

def visualize(image):
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(image)
    plt.savefig("./temp/augmented.jpg")
    #plt.show()
    plt.close()


all_aug = ['CLAHE', 'Blur', 'Cutout', 'GaussNoise', 'HueSaturationValue', 'ChannelShuffle', 'GridDistortion', 'MedianBlur', 'Normalize', 'PadIfNeeded',
           'RandomBrightness', 'RandomBrightnessContrast', 'RandomContrast', 'ToGray', 'ShiftScaleRotate', 'add_rain', 'add_snow', 'add_shadow', 'darken', 'random_brightness']


def applyAugmentation(parameters):
    path = './temp/images/img2.jpg'
    image = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    #images = hp.load_images(path)
    #image = images[0]
    #image *= 255
    image = image.astype(np.uint8)
    transforms = []
    input_aug = parameters["aug_name"]
    i = input_aug

    if i == 'CLAHE':
        clip_limit = int(parameters['clip_limit'])
        tile_grid_x, tile_grid_y = int(parameters['tile_grid_x']), int(parameters['tile_grid_y'])
        tile_grid_size = (tile_grid_x, tile_grid_y)
        prob = float(parameters['prob'])
        print(tile_grid_size)
        transforms.append(A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size, always_apply=False, p=prob))
    elif i == 'Cutout':
        num_holes = int(parameters['num_holes'])
        max_h_size, max_w_size = int(parameters['max_h_size']), int(parameters[max_w_size])
        prob = float(parameters['prob'])
        transforms.append(A.Cutout(num_holes=num_holes, max_h_size=max_h_size,
                                   max_w_size=max_w_size, always_apply=False, p=prob))
    elif i == 'GaussNoise':
        var_limit_x, var_limit_y = int(parameters[var_limit_x]), int(parameters[var_limit_y])
        var_limit = (var_limit_x, var_limit_y)
        prob = float(parameters['prob'])
        transforms.append(A.GaussNoise(var_limit=var_limit,  always_apply=False, p=prob))
    elif i == 'HueSaturationValue':
        hue_shift_limit = int(parameters['hue_shift_limit'])
        sat_shift_limit = int(parameters['sat_shift_limit'])
        val_shift_limit = int(parameters['val_shift_limit'])
        prob = float(parameters['prob'])
        transforms.append(A.HueSaturationValue(hue_shift_limit=hue_shift_limit,
                                               sat_shift_limit=sat_shift_limit, val_shift_limit=val_shift_limit, always_apply=False, p=prob))
    elif i == 'Blur':
        blur_limit = int(parameters['blur_limit'])
        prob = float(parameters['prob'])
        transforms.append(A.Blur(blur_limit=blur_limit, always_apply=False, p=prob))
    elif i == 'ChannelShuffle':
        prob = float(parameters['prob'])
        transforms.append(A.ChannelShuffle(p=prob))
    elif i == 'GridDistortion':
        num_steps = int(parameters[num_steps])
        distort_limit = float(parameters[distort_limit])
        prob = float(parameters['prob'])
        transforms.append(A.GridDistortion(num_steps=num_steps, distort_limit=distort_limit,
                                           interpolation=1, border_mode=4, always_apply=False, p=prob))
    elif i == 'MedianBlur':
        blur_limit = int(parameters['blur_limit'])
        prob = float(parameters['prob'])
        transforms.append(A.MedianBlur(blur_limit=blur_limit, always_apply=False, p=prob))
    elif i == 'Normalize':
        prob = float(parameters['prob'])
        transforms.append(A.Normalize(mean=(0.12, 0.13, 0.14), std=(
            0.668, 0.699, 0.7), max_pixel_value=70, always_apply=False, p=prob))
    elif i == 'PadIfNeeded':
        prob = float(parameters['prob'])
        transforms.append(A.PadIfNeeded(min_height=32, min_width=32,
                                        border_mode=4, value=None,  always_apply=False, p=prob))
    elif i == 'RandomBrightness':
        limit = float(parameters['limit'])
        prob = float(parameters['prob'])
        transforms.append(A.RandomBrightness(limit=limit, always_apply=False, p=prob))
    elif i == 'RandomBrightnessContrast':
        brightness_limit = float(parameters['brightness_limit'])
        contrast_limit = float(parameters['contrast_limit'])
        prob = float(parameters['prob'])
        transforms.append(A.RandomBrightnessContrast(brightness_limit=brightness_limit,
                                                     contrast_limit=contrast_limit, always_apply=False, p=prob))
    elif i == 'RandomContrast':
        limit = float(parameters['limit'])
        prob = float(parameters['prob'])
        transforms.append(A.RandomContrast(limit=limit, always_apply=False, p=prob))
    elif i == 'ToGray':
        prob = float(parameters['prob'])
        transforms.append(A.ToGray(p=prob))
    elif i == 'ShiftScaleRotate':
        shift_limit = float(parameters['shift_limit'])
        rotate_limit = int(parameters['rotate_limit'])
        prob = float(parameters['prob'])
        transforms.append(A.ShiftScaleRotate(shift_limit=shift_limit, scale_limit=0.1,
                                             rotate_limit=rotate_limit, interpolation=1, border_mode=4, always_apply=False, p=prob))
    elif i == 'add_rain':
        rain_type = input(parameters['rain_type'])
        image = am.add_rain(image, rain_type=rain_type, slant=-1, drop_length=1, drop_width=1)
    elif i == 'add_snow':
        snow_coeff = float(parameters['snow_coeff'])
        image = am.add_snow(image, snow_coeff=snow_coeff)
    elif i == 'add_shadow':
        no_of_shadows = int(parameters['no_of_shadows'])
        shadow_dimension = int(parameters['shadow_dimension'])
        image = am.add_shadow(image, no_of_shadows=no_of_shadows, shadow_dimension=shadow_dimension)
    elif i == 'darken':
        darkness_coeff = float(parameters['darkness_coeff'])
        image = am.darken(image, darkness_coeff=darkness_coeff)
    transform = A.Compose(transforms)
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    visualize(transformed_image)
    return transformed_image

# visualize(augmentation(image))
