### Test the classifier
# Test by giving some random images.
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

classifier = load_model('catdog_cnn_model.h5')  # keras.models.Sequential object
test_dir = './test/test/'  # Directory with [1-N].jpg

i_range = range(1,50)  # Will add '.jpg' to each

for i_img in i_range:
    img = test_dir + str(i_img) + '.jpg'
    test_image =image.image_utils.load_img(img, target_size=(64,64)) # A PIL.JpegImagePlugin.JpegImageFile object
    test_image =image.image_utils.img_to_array(test_image) # A np.ndarray object
    test_image =np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    if result[0][0] >= 0.5:
        prediction = 'dog'
        probability = result[0][0]
    else:
        prediction = 'cat'
        probability = 1 - result[0][0]
    
    print('Test Image "' + img + '"')
    print('It is a: ' + prediction.upper() + ' ... ( result = ' + str(round(result[0][0],2)) + ')')
