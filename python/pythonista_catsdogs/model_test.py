### Test the classifier
# Test by giving some random images.
# load_img() selects a random image
#   DOES IT? He said he downloaded images from pexels.com
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

classifier = load_model('catdog_cnn_model.h5')  # keras.engine.sequential.Sequential object
                                                # or perhaps keras.models.Sequential object
test_dir = './test/'  # Directory with [1-N].jpg

i_range = range(1,50)  # Will add '.jpg' to each

for i_img in i_range:
    img = test_dir + str(i_img) + '.jpg'
    test_image =image.image_utils.load_img(img, target_size=(64,64)) # A PIL.JpegImagePlugin.JpegImageFile object
    test_image =image.image_utils.img_to_array(test_image) # A np.ndarray object
    test_image =np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print('result', result)
    if result[0][0] >= 0.5:
        prediction = 'dog'
        probability = result[0][0]
    else:
        prediction = 'cat'
        probability = 1 - result[0][0]
    
    print('Test Image "' + img + '": It is a: ' + prediction
            + ' with ' + str(round(probability*100, 2)) + '% probability')
