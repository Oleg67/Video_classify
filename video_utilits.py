import cv2
import os
import imageio
import numpy as np

import shutil
from sklearn.model_selection import train_test_split

from skimage import filters

print ('OpenCV version', cv2.__version__)

import tensorflow as tf

print ('tf version', tf.__version__)

class Video_Preprocessing(object):
    """
    convert video file  to the image's folders by period , size  and max_length
    if video seperated by a black frame each video will be write in its own folder 
    video_file is the name of video file
    folder_name - main folder name for the file
    period - step with the frame turns to the image
    max_length - max length in frames for the folder of images
    size - if need to resize the image
    video_count - first number of the image's folder
    cut - a flag to cut the original image into x1, x2, y1, y2 image
    mode = {'gray', 'segment', 'lines', 'edges', 'diff', 'orb',  'filter'} is the way to preparation of images
    verbose is the flag of debarging
    
    """
    def __init__(self, 
                 mode='filter',  # method applied to frame
                 max_length=np.inf, # max length of image's folder
                 period=1, # step of frame to the image
                 size=(224,224), # resize the initial image
                 cut=True, # cuttinf the initial image
                 cutting=(400, 1000, 120, 1000), # points of the cutting image
                 verbose=False # flag deburging
                ):
        self.mode = mode
        self.max_length = max_length
        self.period = period
        self.size = size
        self.cut = cut
        self.cutting = cutting
        self.verbose = verbose
        
    def check_make_dir(self, folder):
        """
        check the existing of the folder
        if not make it
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    def video_to_image_folders(self,
                              video_file, # file of video 
                              folder_name, # the main folder that containe daughter's folders for each video
                              video_count # first number of daughter'sfolder 
                             ):
        """
        convert video file  to the image's folders by period , size  and max_length
        if video seperated by a black frame each video will be write in its own folder 
        video_file is the name of video file
        folder_name - main folder name for the file
        """
        image_count = 0 # first image
        
        video = cv2.VideoCapture(video_file)
        while video.isOpened():
            success, image = video.read()
            if not success:
                print ('No read may be incorect the video file or the name, or the end of the video')
                break
            
            if np.any(image == None):
                continue
            # simply cut the most important image segment
            if self.cut:
                x1, x2, y1, y2 = self.cutting
                image = image[x1:x2, y1:y2]
                
        
            # check the black seperating 
            black = np.all(image <= 0.01)
            # seperate by black   
            if black:
                # if there is a black image turn to the next folder 
                if image_count != 0:
                    video_count += 1
                image_count = 0
                continue
                
            else:
                # if not seperating creat the file to write in the same folder
                if not os.path.exists(folder_name +'/video_' + str(video_count).zfill(5)):
                    os.makedirs(folder_name +'/video_' + str(video_count).zfill(5))
                                
                # if yes seperating creat the new folder
                else:
                    # check the length of the folder 
                    if len(os.listdir(folder_name +'/video_' + str(video_count).zfill(5))) >= self.max_length:
                        video_count += 1
                        image_count = 0 
                        os.makedirs(folder_name +'/video_' + str(video_count).zfill(5))
       
                # period chosen the image
                #print image_count, image_count% self.period
                if not (image_count% self.period == 0): # use only images with the period
                    image_count += 1
                    continue
                
                filename = (folder_name +'/video_' +str(video_count).zfill(5) + '/' 
                                  + str(image_count+1).zfill(6) +'_image.jpg' )              
                
                # HSV represantation of the imade
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # gray presantation of image
                if self.mode =='gray':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            
                # find lines on the image and add lines to the image
                elif self.mode =='lines':
                    edges = cv2.Canny(image, 4, 70, apertureSize=3)
                    lines = cv2.HoughLines(edges, 1, np.pi/180, 200, 100, 10)
                    
                    for xx in lines:
                        rho, theta = xx[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000* (a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000* (a))
                            
                        # use lines have only the angle
                        if (theta > 0.7) & (theta < 1.8):
                            image = cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                               
                # find contours on the image and turn the image in this style
                elif self.mode =='edges':
                    image = cv2.Canny(image, 60, 100, apertureSize=3)
                                        
                # find thresholds on the image and turn the image in this style    
                elif self.mode =='thresh':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.medianBlur(image, 5)
                    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                                      cv2.THRESH_BINARY, 13, 4)
                
                # find background on the video and diff the backgrounf from the image  
                elif self.mode =='segment':
                    fgbg = cv2.createBackgroundSubtractorMOG2()
                    image = fgbg.apply(image)
                        
                # find the difference between two serial images
                elif self.mode =='diff':
                    if image_count == 0:
                        image_old = image.copy()
                    image_diff = cv2.bitwise_xor(image, image_old)
                    image_old = image.copy()
                    image = cv2.cvtColor(image_diff, cv2.COLOR_BGR2GRAY)
                     
                # find the org poins in images
                elif self.mode =='orb':
                    # Initiate STAR detector
                    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)

                    # find the keypoints with ORB
                    kp = orb.detect(image,None)

                    # compute the descriptors with ORB
                    kp, des = orb.compute(image, kp)

                    # draw only keypoints location,not size and orientation
                    image = cv2.drawKeypoints(np.zeros_like(image),kp, None, color=(255,255,255), flags=cv2.DrawMatchesFlags_DEFAULT)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
                # apply the filter to the images
                elif self.mode =='filter':
                    #filter parameters
                    blue_low = np.array([20, 30, 100])
                    blue_high = np.array([160, 250, 255])
                    # Threshold the HSV image to get only blue colors
                    mask = cv2.inRange(image, blue_low, blue_high)

                    # Bitwise-AND mask and original image
                    image = cv2.bitwise_and(image,image, mask= mask)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
                #resize image
                if self.size is not None:
                    image = cv2.resize(image, self.size, interpolation=cv2.INTER_NEAREST)
                                
                if self.verbose:
                    print (filename, '  ',image.shape)
                
                # write the image in file   
                try:
                    imageio.imwrite(filename, image)
                except ValueError:
                    continue

                image_count +=1
    
        numbers_of_video = np.zeros(video_count, dtype=np.int32)
    
        # count the number of images that wrote in folders as a file
        for i,f in enumerate(os.listdir(folder_name)):
            numbers_of_video[i] = len(os.listdir(folder_name +'/' + f))
   
        print ('*** Done *** write in ', folder_name)
        print ('model of preprocession is, ', self.mode)
        print ('number of videos ', len(numbers_of_video))
        print ('max_image_len ', int(numbers_of_video.max()))
        print ('min_image_len ', int(numbers_of_video.min()))
        
    def video_folder_to_image_folders(self,
                                       video_folder, # folder of video or file
                                       folder_name=None, # name of main image' folder
                                      ):
        """
        convert a video folder with the same class of video to image's folders
        each video will be convert to its own daughter's image's folder
        video_folder - the folder contains the video files the same class
        folder_name - name of image's folders as rule the name of the class
        """
        if folder_name is None:
            if os.path.isdir(video_folder):
                folder_name = video_folder + '_images'
            elif os.path.isfile(video_folder):
                folder_name = video_folder.split('.')[0] + '_images'
        
    
        if os.path.isfile(video_folder):
            video_list = [os.path.join(os.getcwd(), video_folder)]
        elif os.path.isdir(video_folder):
            video_list = list(map(lambda x: os.path.join(os.getcwd(), video_folder, x), os.listdir(video_folder)))
        else:
            print (video_folder, 'no folder nor file')
            return
    
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            n = 1
        else:
            n = len(os.listdir(folder_name)) + 1
    
        for video_file in video_list:
            self.video_to_image_folders(video_file, folder_name, n)
            n = len(os.listdir(folder_name)) + 1
    
    def different_video_folders_to_image_folders(self,
                        video_folders, # folder contain the video of different classes
                        folders_names=None # the list of clesses names
                        ):
        """
        convert folders of video with different classes to foldes of images
        video_folders - the folders contain the video folders or files of different classes
        folder_names - names of image's folders as rule the names of classes, the same length as the video folders
        """
        if not os.path.isdir(video_folders):
            print (video_folders, ' is not a video folders')
            print ('get the video folders')
            return
        else:
            video_folders_list = sorted(list(map(lambda x: os.path.join(os.getcwd(), video_folders, x), 
                                     os.listdir(video_folders))))
        
        _folders_names = []
        for i, video_folder in enumerate(video_folders_list):
            if not folders_names:
                folder_name = video_folders + '_images/' +'class_'+str(i)
            else:
                assert len(folders_names) == len(video_folders_list), 'length of video folders is not equal to the name foldes'
                folder_name = folders_names[i]
        
            self.video_folder_to_image_folders(video_folder, folder_name)
            folder_name = os.path.join(ob.getcwd(), folder_name)
            _folders_names.append(folder_name)
            
        self.foldes = _folders_names
        print ('write the images in the ',_folders_names)
        
        

class Images_Preprocessing(Video_Preprocessing):
    """
    serve to preprocessing sets of images to create a train and a test folders of image's folders
    may use as video preprocesing to convert the video file  to the image's folders by period , 
    size  and max_length
    if video seperated by a black frame each video will be write in its own folder 
    video_file is the name of video file
    in this case needs the video_files or video_foldes
    """

    def __init__(self, folders=None, **kwards):
        super(Image_preprocesing, self).__init__()
        self.folders = folders # the folder contains image's folder gor each class
        #redefine parameters of the class
        for k in kwards.keys():
            self.__dict__[k] = kwards[k]
        

    
    
    def copy_files_to_folder(self, list_files, folder):
        """
        copy the list of files in the folder
        """
        # check the existing of the folder if not make
        if not os.path.exists(os.path.join(os.getcwd(),folder)):
            os.makedirs(os.path.join(os.getcwd(), folder))
        
        lable = os.path.split(folder)[-1]
        for f in list_files:
            _file = os.path.basename(f)
            _dir = os.path.split(os.path.dirname(f))[-1]
            if _dir != lable:
                if not os.path.exists(os.path.join(os.getcwd(), folder, _dir)):
                    os.makedirs(os.path.join(os.getcwd(), folder, _dir))
                    
                shutil.copyfile(f , os.path.join(os.getcwd(), folder, _dir, _file))
            else:
                shutil.copyfile(f , os.path.join(os.getcwd(), folder, _file))
            
    
    def create_train_test_folders(self, video_folder=None, test_size=0.15):
        """
        seperate images on train and test foldes 
        list images - list of image' paths 
        test_size is the size of the test folder 
        """
        if self.folders is None:
            try:
                self.different_video_folders_to_image_folders(video_folder)
            except TypeError:
                print ('need the video folder')
        else:
            folder = os.path.dirname(self.folders[0])
        
        list_images = sorted(self.get_list_of_images_path(folder))
        _list_images = list(map(lambda x: x.split('/'), list_images))
        
        mask = np.array(list(map(lambda x: not ('train' in x ) or ('test' in x), _list_images)))
        list_images = np.array(list_images)[mask]
        y = np.vectorize(lambda x: x.split('/')[-3])(list_images)
    
        images_train, images_test, y_train, y_test = train_test_split(list_images, y, stratify=y, 
                                                                      test_size=test_size)
    
        for fd, img, y in zip(['train', 'test'], (images_train, images_test), (y_train, y_test)):
            for lable in np.unique(y):
                self.copy_files_to_folder(img[y == lable], os.path.join(folder, fd, lable))
                print ('write the folder', os.path.join(folder, fd, lable))
                
    
    
    def get_list_of_images_path(self, folder):
        """
        create a list of all images' paths  for all subfolders in the folder
        """
        fd_list = []
    
        if not self.all_dir(folder):
        # add to the list only files not dir
            return sorted(list(map(lambda x: os.path.join(folder, x), os.listdir(folder))))
    
        for fd in os.listdir(folder):
            path = os.path.join(os.getcwd(), folder, fd)
            fd_list += self.get_list_of_images_path(path)

        return fd_list 

    def all_dir(self, folder):
        """
        serve to check all filenames in folder is dir
        """
        flag = True
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                flag = flag & os.path.isdir(os.path.join(folder,f))
            
        else:
            flag = False
        
        return flag    


class Images_to_sequence(Video_Preprocessing):
    """
    to convert image's files to image's sequence  and feature extracting
    folders - folder contains the folders of images
    max_sequence - the maximum length of sequence
    period_sequence - period of images add to the sequnce
    """
    def __init__(self, 
                 folders=None, # list of image's folders
                 max_sequence=20, # max length of sequence
                 period_sequence=5, # period of images to add to the sequence
                **kwards): 
        super(Images_to_sequence, self).__init__()
        self.max_sequence = max_sequence
        self.period_sequence = period_sequence
        self.folders = folders
        #redefine parameters of the class
        for k in kwards.keys():
            self.__dict__[k] = kwards[k]
    
    def get_folders(self):
        """
        check the folders are in the same folder
        """
        if not self.folders is None:
            
            if len(set(map(os.path.dirname, self.folders))) !=1:
                raise OSError('the folders of classes are not in the same folder')
            if len(list(map(os.path.split, self.folders))) == 1:
                _dif = self.folders[0]
                _dir_list = list(map(lambda x: os.path.join(_dir, x), os.listdir(_dir)))
                mask = list(map(os.path.isdir, _dir_list))
                _dir_list = np.array(_dir_list)[mask]
                _dir_list = _dir_list[(_dir_list != 'train') & (_dir_list != 'test')]
                self.folders = list(map(lambda x: os.path.join(_dir, x), _dir_list))
                return
        pass
                
            
            
            
    def create_dict_images_lists(self, folders=None):
        """
        create the dictionary of each kyes as a list of image's path
        """
        
        self.folders = folders or None
        if self.folders is None:
            print ('need the folder with images')
            return
        f_lists = {}
        
        for f in self.folders:
            f_lists[f] = self.get_list_sequence_images(f) # make the dictionary of images' lists
            
        self.f_lists = f_lists
        
            

    def get_list_sequence_images(self, path):
        """
        serve to make the 2D tensor from the list of images
        with max length and period
        return 2D tensor (number of images' sequences, max length)
        """
        #print 'do get_list_sequence_images'
        images_list = np.vectorize(lambda x: os.path.join(os.getcwd(), path, x))(np.sort(os.listdir(path)))
        files_list = np.array([])
        
        for _folder in images_list:
            #print _folder
            files_list = self.create_list_of_images(_folder, files_list)
            #print files_list.shape
            
        return files_list.reshape(len(files_list)//self.max_sequence, self.max_sequence)
    
    def add_list_length(self, f_list, max_length):
        """
        make the list the max_lenght
        """
        f_list = np.array(f_list) 
        add_length = max_length - len(f_list)
        #print 'DO add_list_length add_length', add_length
        if add_length ==0:
            return f_list
        elif add_length < 0:
            n = max_length * (len(f_list)// max_length)
            #print f_list[n:]
            return np.append(f_list[:n], self.add_list_length(f_list[n:], max_length))
        f_temp = np.random.choice(f_list, size=add_length)
        #print type(f_temp)
        
        return np.sort(np.append(f_list, f_temp))



    def lists_by_period(self, f_list, period, max_length):
        """
        create lists max length by period with shift 
        """
        mask = np.arange(len(f_list)) % period == 0
        f_temp = np.array(f_list)[mask]
        f_temp = self.add_list_length(f_temp, max_length)
        
        for j in range(1, period):
            mask[-1] = False
            mask = np.roll(mask, 1) # shift the period' mask to one step
            f_temp2 = np.array(f_list)[mask]
            f_temp2 = self.add_list_length(f_temp2, max_length)
            f_temp = np.append(f_temp, f_temp2)
            
        return f_temp
     
            
    def create_list_of_images(self, folder, f_lists):
        """
        serve to make the list of video images as a sequence with the period 
        max_length  - the length of images' sequence
        return list of images as sequence in  a list of paths
        """
        #print 'DO create_list_of_images'
        #print('folder', folder)
        f_list = sorted(os.listdir(folder))
        #print(f_list)
        f_list = list(map(lambda x: os.path.join(os.getcwd(), folder, x), f_list)) # the full names of files
        f_list = np.array(f_list)
        #print f_lists
        
        #print ('shape', f_list.shape)
        #print (len(f_list))
        
        ind_mask = (np.arange(len(f_list)) % self.period_sequence == 0) # mask for images with the period
        
        # if the number of images in the folder less then max length 
        # reduce the period or fill the sequence the same images
        period = self.period_sequence
        while (np.sum(ind_mask) < self.max_sequence) & (period > 1):
            #print 'do reduce the period', period
            period -=1
            ind_mask = np.arange(len(f_list)) % period == 0
        else:
            #print '*******'
            return np.append(f_lists, self.add_list_length(f_list, self.max_sequence))
        
        # if the number of images in the folder  more then max length
        # images from folder are taken to create the image's sequence with the max length by shift the mask 
        f_list = np.array(f_list)[ind_mask]
        #print f_list
        f_lists = np.append(f_lists, self.lists_by_period(f_list, period, self.max_sequence))
        
        return f_lists
        
        
    def store_images_features(self, cnn_Model):
        """
        store the image's features in .ny file
        cnn_Model - cnn model to extract   features from images
        """
        self.create_dict_images_lists()
        
        for k in self.f_lists.keys():
            k = os.path.join(os.getcwd(), k)
            _file = os.path.split(k)[0] + '/X_' + os.path.split(k)[1]
            X = self.create_tensor_of_features(k, cnn_Model)
            np.save(_file, X)
            print ('save X', X.shape, 'in file', _file, '.npy')
    
        
    def create_tensor_of_features(self, key, model):
        """
        from the 2D tensor of images make 3D the data's features tensor using  the model to extract image's features
        image_list - 2D tensor of images' names that were united to make images' sequences
        model - the model to extract the image' features
        return the 3D tensor (number of images, length of images' sequence, number of image' features)    
        """
        image_list = self.f_lists[key]
        N = image_list.shape[0]
        l = image_list.shape[1]
        n, = model.output_shape[1:]
    
    
        X = np.zeros((N, l, n))
    
        for i in range(N):
            x = self.create_tensor_from_list_of_image_sequence(image_list[i])
            #print x.shape
            features = model.predict_on_batch(x[:, :, :, np.newaxis])
        
            #features = tf.nn.batch_normalization(features, )
            X[i] = features.reshape(features.shape[0], features.shape[1])
        
        return X
    
        
            
    def create_tensor_from_list_of_image_sequence(self, image_list):
        """
        get the tensor from lists of images
        """
        size = self.get_image_size()
    
        # chech How much chanal does the image have?
        if (len(size) == 2) or (size[2] ==1):
            w, h, c = size[0], size[1], 1
            X = np.full((len(image_list), w , h ), 0, dtype=np.int8)
        elif len(size) == 3 :
            w, h, c = size[0], size[1], 3
            X = np.full((len(image_list), w , h, c), 0, dtype=np.int8)

        for i,img in enumerate(image_list):
            image = tf.keras.preprocessing.image.load_img(img, target_size=size)
            if c == 1:
                X[i] = tf.keras.preprocessing.image.img_to_array(image)[:,:,0]
            elif c ==3:
                X[i] = tf.keras.preprocessing.image.img_to_array(image)
         
        return X

    def get_image_size(self):
        """
        return the size of images in the folder
        """
        try:
            path = os.path.join(os.getcwd(), self.folders[0])
        
            while os.path.isdir(path):
                filename = os.listdir(path)[0]
                path = os.path.join(path, filename)
            
        except OSError as e:
            print (str(e))
            return

        return imageio.imread(path).shape



