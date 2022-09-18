import numpy as np
import time
import random


IMAGE_KEY = 'image'
BINARY_KEY = 'binary'
CLASSIFICATION_KEY = 'classification'
LOCALIZATION_KEY = 'localization'


def get_info_tuple_from_params(frame_number=0, camera_number=0, split_number=0):
    """this function is used to get tuple of image split infoes (frame_number, camera_number, split_number) from inputs

    :param frame_number: _description_, defaults to 0
    :type frame_number: int, optional
    :param camera_number: _description_, defaults to 0
    :type camera_number: int, optional
    :param split_number: _description_, defaults to 0
    :type split_number: int, optional
    :return: _description_
    :rtype: _type_
    """

    return (frame_number, camera_number, split_number)

    
def get_params_from_info_tuple(info_dict):
    """this function is used to get image split info params from its info_tuple

    :param info_dict: _description_
    :type info_dict: _type_
    :return: _description_
    :rtype: _type_
    """

    frame_number = info_dict[0]
    camera_number = info_dict[1]
    split_number = info_dict[2]

    return frame_number, camera_number, split_number



class List_Management():
    """this class is used to creat a dict object of image splits,
    in this dict, each split has a key of split info that is a tuple in form of (frame_number, camera_number, split_number),
    and the key is a dict in form of {image: np.ndarray, binary:0or1, localization:[], classification:[]}
    """

    def __init__(self, batch_size=128, image_shape=(300,300,3), sensitive_safety=False, lock_pop=False):
        """init function of class
        Parameters
        ----------
        batch_size : int
            quantity of batch size
        safety_mode : str, optional
            string indicate the sensitivity of class to miss data, by default 'normal'
            is for indicating the sensitivity of class to miss data
            miss data is----> image without tuple,tuple without tuple,and size of is image is not normal
            safety_mode kind --->none/normal/sensitive
        img_shape : tuple, optional
            tuple consists of length and width of image,that will appen to the list, by default (1900,1600)
        """

        self.batch_size = batch_size  # batch size of output list
        self.sensitive_safety = sensitive_safety  # indicate the safeness & sensitivity of class for checking miss data
        self.image_shape = image_shape  # tuple of Length and width of normal image
        self.images_list = {}  # dict of info-image pair
        self.len_list = 0  # length of dict of image
        self.n_available_batches = 0 # number of prepared batch 
        self.lock_pop = lock_pop


    def update_n_available_batches_and_len_list(self):
        """this function is used to update number of available batches

        :param number_of_batch: _description_
        :type number_of_batch: _type_
        """

        self.len_list = len(self.images_list)
        self.n_available_batches = (self.len_list // self.batch_size) 


    def set_image_shape(self, image_shape):
        """set function of img_shape attribute (n_rows, n_cols, n_channel(if needed))

        Parameters
        ----------
        img_shape : tuple
            tuple of Length and width of normal image
        """

        self.image_shape = image_shape

    
    def set_batch_size(self, batch_size):
        """this function is used to set batch_size

        :param batch_size: _description_
        :type batch_size: _type_
        """

        self.batch_size = batch_size


    def set_sensitive_safety(self, enable=True):
        """this function is used to enable/diable sensivity mode

        :param enable: _description_
        :type enable: boolean
        """

        self.sensitive_safety = enable
    

    def set_lock_pop(self, enable=True):
        """this functiopn is used to enable or disable lock pop

        :param enable: _description_
        :type enable: boolean
        """

        self.lock_pop = enable

    
    def clear_list(self):
        """this function is used to clear all list contents
        """

        self.images_list.clear()
        self.update_n_available_batches_and_len_list()


    def insert_image_split(self, split_info, split_value, reject_duplicate=True):
        """this functuion is used ti to insert a new image split and its info to list,
        split_info that is a tuple in form of (frame_number, camera_number, split_number),
        split_value: dict in form of {image: np.ndarray, binary:0or1, localization:[], classification:[]}

        :param split_info: _description_
        :type split_info: _type_
        :param image_split: _description_
        :type image_split: _type_
        """
        
        while split_info in self.images_list.keys():
            if (split_info[0] is None or split_info[1] is None) and not reject_duplicate:
                split_info = list(split_info)
                split_info[-1] += 1
                split_info = tuple(split_info)

            else:
                return
        # check image to be in right format
        if self.sensitive_safety:
            if split_value[IMAGE_KEY] is not None and split_value[IMAGE_KEY].shape==self.image_shape:
                self.images_list[split_info] = split_value

        else:
            self.images_list[split_info] = split_value

        # update number of available batches
        self.update_n_available_batches_and_len_list()


    def get_batch(self, force=False):
        """this function is used to get a batch,
        if force=True, get a batch also if len list is smaller than a batch (with zero pad),
        if pop=True, pop and get the batch

        :param force: _description_, defaults to False
        :type force: bool, optional
        :param pop: _description_, defaults to False
        :type pop: bool, optional
        :return: _description_
        :rtype: _type_
        """

        # prevent pop if lock_pop is enabled
        split_infoes = []
        split_values = []
        split_images = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2])).astype(np.float32)

        if self.len_list>0:
            # if force and len list smaller than a batch, zero append and get batch
            if force:
                # checking there is data less than batch_size in the lists
                if self.len_list < self.batch_size:
                    # if there is less data less than batch_size in the lists
                    # this code add some miss data to the list for setting len of list to batch_size
                    temp_split_value = {IMAGE_KEY:np.zeros((self.image_shape[0],self.image_shape[1],self.image_shape[2]), dtype='float32'), BINARY_KEY:None, LOCALIZATION_KEY:None, CLASSIFICATION_KEY:None}
                    temp_split_info = (None,None,0)  # info_tuple

                    for _ in range(self.batch_size - self.len_list):
                        self.insert_image_split(split_info=temp_split_info, split_value=temp_split_value, reject_duplicate=False)

                while len(split_infoes)<self.batch_size and len(split_values)<self.batch_size:
                    try:
                        dict_item = (di := next(iter(self.images_list)), self.images_list.pop(di))
                        self.update_n_available_batches_and_len_list()
                        split_infoes.append(dict_item[0])
                        split_values.append(dict_item[1])
                        split_images[len(split_infoes)-1] = dict_item[1][IMAGE_KEY]
                
                    except:
                        pass

                return split_infoes, split_values, split_images

            # normal mode, if len list smaller than batch, return None
            else:
                if self.len_list > self.batch_size:
                    while len(split_infoes)<self.batch_size and len(split_values)<self.batch_size:
                        try:
                            dict_item = (di := next(iter(self.images_list)), self.images_list.pop(di))
                            self.update_n_available_batches_and_len_list()
                            split_infoes.append(dict_item[0])
                            split_values.append(dict_item[1])
                            split_images[len(split_infoes)-1] = dict_item[1][IMAGE_KEY]
                    
                        except:
                            pass

                    return split_infoes, split_values, split_images

                else:
                    return None, None, None

        # list empty
        else:
            return None, None, None



if __name__=='__main__':


    lm_obj = List_Management()
    lm_obj.set_sensitive_safety(enable=True)

    start = time.time()

    for i in range(10000):

        start = time.time()
        fr_num = None # random.randint(0, 100)
        ca_num = random.randint(0, 24)
        split_num = random.randint(0, 12)
        img = np.zeros((300,300,3), dtype='float32')

        lm_obj.insert_image_split(split_info=(fr_num,ca_num,split_num), split_value={IMAGE_KEY:img, BINARY_KEY:None, LOCALIZATION_KEY:None, CLASSIFICATION_KEY:None})
    
    print('Elapsed time: %s' % (time.time()-start))
    # print(lm_obj.len_list)

    # print(list(lm_obj.images_list.values())[0])
        
    
    start = time.time()


    while lm_obj.len_list>0:
        

        split_infoes, split_values = lm_obj.get_batch(force=True, pop=True)
        
        split_images = [splv[IMAGE_KEY] for splv in split_values]
        split_images = np.asarray(split_images)
        

        # for splv in split_values:
            
        #     splv[BINARY_KEY] = 1
        #     print(splv)
        #     pass

    
    print('Elapsed time: %s' % (time.time()-start))
