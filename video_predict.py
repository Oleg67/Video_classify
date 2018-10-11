import cv2
import skvideo
import math
import os
import imageio
import numpy as np

import shutil
from sklearn.model_selection import train_test_split

from skimage import filters
from video_utilits import Images_to_sequence, Video_Preprocessing

import tensorflow as tf
import tflearn


print ('openCV version',cv2.__version__)
print ('tf version',tf.__version__)


class Video_Prediction(Images_to_sequence):
    """
    predict the video class by cnn_Model and rnn_Model
    """
    def __init__(self,
                 cnn_Model, # cnn model to extract the features from the image
                 rnn_Model, # rnn model to predict class of the video
                 # max_length=20, # max length of the image's sequence
                 period=5, # period with then images are added to sequence
                 **kwards):
        super(Video_Prediction, self).__init__()
        self.cnn_Model = cnn_Model
        self.rnn_Model = rnn_Model
        self.period = period
        
        self.max_length = rnn_Model.input_shape[1] # max length of the image's sequence
        self.max_sequence = self.max_length # max length of the image's sequence
        
        self.period_sequence = self.period_sequence // period if period < self.period_sequence else 1
        self.size = cnn_Model.input_shape[1:3][::-1] # size of images to write and read 
        
        #redefine parameters of the class
        for k in kwards.keys():
            self.__dict__[k] = kwards[k]
            
    
        
    def predict(self, video_file):
        """
        classify of the video_file by rnn_Model and cnn_Model
        """
        if os.path.exists('prediction'):
            shutil.rmtree('prediction') # del the folder if one exists
        
        self.video_to_image_folders(video_file, 'prediction', 1) # preprocessing of video file to image's foldes
        self.create_dict_images_lists(['prediction']) # images to sequence 
        X = self.store_images_features()
        
        return self.rnn_Model.predict_on_batch(X)
    
    def store_images_features(self):
        """
        store the image's features in .ny file
        cnn_Model - cnn model to extract   features from images
        """
        #self.create_dict_images_lists()
        
        for k in self.f_lists.keys():
            _dir = os.path.join(os.getcwd(), k)
            _file = os.path.split(_dir)[0] + '/X_' + os.path.split(_dir)[1]
            #print _file
            X = self.create_tensor_of_features(k, self.cnn_Model)
            np.save(_file, X)
            print ('save X', X.shape, 'in file', _file +'.npy')
        return X
        
if __name__ == '__main__':
    cnn_model = tf.keras.models.load_model('cnn_model.h5')
    cnn_Model = tf.keras.Model(inputs=cnn_model.input,
                   outputs=cnn_model.get_layer(index=-3).output)

    rnn_model = tf.keras.models.load_model('video_clasification2.h5')
    print ('cnn input', cnn_Model.input_shape, 'cnn output', cnn_Model.output_shape)
    print ('rnn input', rnn_model.input_shape, 'rnn output', rnn_model.output_shape)
    print
    print 'enter the video file name'
    
    name_video = input()
    
    cv = Video_Prediction(cnn_Model, rnn_model, period=5)
    cv.predict(name_video)
