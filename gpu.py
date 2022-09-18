import threading
import time
import numpy as np
import tensorrt as trt
from PySide6.QtCore import Signal as sSignal
from PySide6.QtCore import QObject as sQObject

import inference
import list_management_dict as lm


HEIGHT = 224
WIDTH = 224
BATCHSIZE = 128
NCLASSES = 1000


class GPU_Handler(sQObject):
    """this class is used to handel gpu and run inference on inputs
    """

    # signal for updating ui
    fps_progress_signal = sSignal(str, int)

    def assign_parameters(self, gpu_idx, thread_idx, engine_path, batch_size, input_width, input_height, n_channel, n_classes, defect_list_obj, perfect_list_obj):
        """this function is used to create gpu handler object and set needed objects and parametrs

        :param gpu_idx: _description_
        :type gpu_idx: _type_
        :param thread_idx: _description_
        :type thread_idx: _type_
        :param engine_path: _description_
        :type engine_path: _type_
        :param batch_size: _description_
        :type batch_size: _type_
        :param input_width: _description_
        :type input_width: _type_
        :param input_height: _description_
        :type input_height: _type_
        :param n_channel: _description_
        :type n_channel: _type_
        :param defect_list_obj: _description_
        :type defect_list_obj: _type_
        :param perfect_list_obj: _description_
        :type perfect_list_obj: _type_
        """
        
        # define gpu index and index of thread to run on gpu
        self.gpu_idx = gpu_idx
        self.thread_idx = thread_idx
        self.engine_path = engine_path

        # input and output storage for engine model (with size one, one batch each time)
        self.input_split_value = None
        self.input_split_images = None
        self.input_split_info = None
        self.defect_list_obj = defect_list_obj
        self.perfect_list_obj = perfect_list_obj

        # input parameters
        self.batch_size = batch_size
        self.input_width = input_width
        self.input_height = input_height
        self.n_channel = n_channel
        self.n_classes = n_classes

        # create model inference object from engin path
        self.inference_obj = inference.Inference(engine_path=self.engine_path, cuda_idx=self.gpu_idx)
        self.inference_obj.allocate_buffers(self.batch_size, trt.float32)
        self.fps = 0

    
    def set_input_batch(self, input_split_infoes_batch, input_split_values_batch, input_split_image_batch):
        """this function is used to set new  input batch to gpu object for running inference

        :param input_split_infoes_batch: _description_
        :type input_split_infoes_batch: _type_
        :param input_split_values_batch: _description_
        :type input_split_values_batch: _type_
        :return: _description_
        :rtype: _type_
        """

        
        # check if input storage is empty
        if self.input_split_info is None and self.input_split_value is None and self.input_split_images is None:
            # 
            self.input_split_info = input_split_infoes_batch
            self.input_split_value = input_split_values_batch
            # extract images from dicts in batch

            # t1 = time.time()
            self.input_split_images = input_split_image_batch

            # print('set input', time.time()-t1)

            return True

        # input full
        return False
    

    def pred_to_class_id(self, pred):
        """this function is used to translate model output probs to class id

        :param pred: _description_
        :type pred: _type_
        :return: _description_
        :rtype: _type_
        """

        if len(pred.shape)==1:
            pred = np.reshape(pred, (self.batch_size, self.n_classes))
            
        return np.argmax(pred, axis=1)
    

    def destroy_object(self):
        """this function is used to destroy inference objects
        """

        try:
            self.inference_obj.destory()
        except:
            pass

    
    def clear_inputs(self):
        """this function is used to clear gpu inputs
        """
        # empty inputs
        self.input_split_images = None
        self.input_split_info = None
        self.input_split_value = None

    
    def get_fps(self):
        """this function is used to test gpu fps on model
        """

        # test image
        image = np.random.rand(self.batch_size, self.n_channel, self.input_height, self.input_width).astype(np.float32)

        # run inference
        start = time.time()
        _ = self.inference_obj.do_inference(pics_1=image, batch_size=self.batch_size,
                                                        height=self.input_height, width=self.input_width)

        self.fps =  int(self.batch_size/(time.time()-start))
        return self.fps
    

    def gray_to_rgb(self, input_batch):
        """this function is used to convert 1 channel image to 3 channel

        :param input_batch: _description_
        :type input_batch: _type_
        :return: _description_
        :rtype: _type_
        """

        if input_batch.shape[-1] == 1:
            input_batch = input_batch[:,:,:,0]
            input_batch = np.repeat(input_batch[:, :, :, np.newaxis], self.n_channel, axis=3)

        elif input_batch.shape[1] == 1:
            input_batch = input_batch[:, 0]
            input_batch = np.repeat(input_batch[:, :, :, np.newaxis], self.n_channel, axis=3)

        return input_batch


    def run_inference_binary(self):
        """this function is used to run inference on model input on a loop
        """

        # run model if any input available
        if self.input_split_images is not None:
            # start = time.time()

            # run inference
            model_output = self.inference_obj.do_inference(pics_1=self.input_split_images, batch_size=self.batch_size,
                                                            height=self.input_height, width=self.input_width)

            # change model output probs to class index
            binary_class_ids = self.pred_to_class_id(pred=model_output)

            # insert outputs to defect and perfect lists
            self.insert_binary_results_to_defect_perfect_lists(binary_class_ids=binary_class_ids)
            
            # empty inputs
            self.input_split_images = None
            self.input_split_info = None
            self.input_split_value = None

            # fps = self.batch_size/(time.time()-start)

            # print('gpu ids %s, fps: %s' % (self.gpu_idx, fps))

            return True
        
        else:
            return False

    
    def insert_binary_results_to_defect_perfect_lists(self, binary_class_ids):
        """this class is used to insert output results from binary model to defect and perfect lists

        :param binary_class_ids: _description_
        :type binary_class_ids: _type_
        """

        for cls_id, splv, splf in zip(binary_class_ids, self.input_split_value, self.input_split_info):
            # set binary results on each split value
            splv[lm.BINARY_KEY] = cls_id

            # add to perfect list
            if cls_id == 0:
                self.perfect_list_obj.insert_image_split(split_info=splf, split_value=splv)
            
            # add to defect list
            else:
                self.defect_list_obj.insert_image_split(split_info=splf, split_value=splv)


# def test_binary_fps(gpu_objects_list, test_time=10):
#     """this function is used to test gpu fps on model
#     """

#     for gpu in gpu_objects_list:

#         # animation increasing round progressbar first
#         # get fps
#         fps = gpu.get_fps()

#         # set on ui prigress by signal
#         gpu.fps_progress_signal.emit('binary_gpu%s_fps_rpb' % (gpu.gpu_idx), 0)
#         for i in range(0, fps, 50):
#             gpu.fps_progress_signal.emit('binary_gpu%s_fps_rpb' % (gpu.gpu_idx), i)
#             time.sleep(0.01)
        
#         # main process
#         start_time = time.time()
#         t1 = time.time()

#         while time.time()-start_time<=test_time:
#             # get fps
#             fps = gpu.get_fps()

#             # set on ui prigress by signal
#             if time.time()-t1>=0.2:
#                 gpu.fps_progress_signal.emit('binary_gpu%s_fps_rpb' % (gpu.gpu_idx), fps)
#                 t1 = time.time()

#     # test multi gpu
#     # animation increasing round progressbar first
#     # get fps
#     fps = 0
#     for gpu in gpu_objects_list:
#         fps += gpu.get_fps()
#     fps = int(fps*0.8)

#         # set on ui prigress by signal
#         gpu.fps_progress_signal.emit('binary_gpu%s_fps_rpb' % (gpu.gpu_idx), 0)
#         for i in range(0, fps, 50):
#             gpu.fps_progress_signal.emit('binary_gpu%s_fps_rpb' % (gpu.gpu_idx), i)
#             time.sleep(0.01)

#         threads_list = []
        
#         for gpu in gpu_objects_list:
#             gpu.fps = 0
#             print(gpu.fps)
#             threads_list.append(threading.Thread(target=gpu.get_fps))

#         start = time.time()

#         for thrd in threads_list:
#             thrd.start()
            
#         fps = int(gpu.batch_size/(time.time()-start))

#         print(fps, gpu.fps)
        

        




if __name__=='__main__':
    
    # outputs
    defect_list_obj = lm.List_Management()
    perfect_list_obj = lm.List_Management()

    gpu_obj_0 = GPU_Handler(gpu_idx=0, thread_idx=0, engine_path="../resnet_cnn_0.plan",
                            batch_size=BATCHSIZE, input_width=WIDTH, input_height=HEIGHT, n_channel=3, n_classes=NCLASSES,
                            defect_list_obj=defect_list_obj, perfect_list_obj=perfect_list_obj)


    # inputs
    input_split_infoes_batch = [(None,None,0) for i in range(128)]
    input_split_values_batch = [{lm.IMAGE_KEY:np.zeros((224,224,3), dtype='float32'), lm.BINARY_KEY:None} for i in range(128)]

    gpu_obj_0.set_input_batch(input_split_infoes_batch=input_split_infoes_batch, input_split_values_batch=input_split_values_batch)

    


    
    def run_inference():
        while True:
            start = time.time()
            gpu_obj_0.run_inference_binary()

            print('FPS: %s' % str(BATCHSIZE/(time.time()-start)), perfect_list_obj.len_list)

            time.sleep(0.1)

        
    threading.Thread(target=run_inference).start()
