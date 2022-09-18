import time
import numpy as np
import cv2
from pathStructure import create_sheet_path, sheet_image_path
import threading
from split import ImageCrops_3d
import list_management_dict as lm

class data_grabber():
    def __init__(self, cameras, n_grab_threads, main_path, split_size, input_list):
        self.cameras = cameras
        self.n_threads = n_grab_threads
        self.read_threads = []
        self.sheet_id = 0
        self.coil_dict = {}
        self.image_format = '.png'
        self.nframe = [0] * 24
        self.start_cam = 1
        self.stop_cam = 0
        self.main_path = main_path
        self.frame_update_time = 5
        self.stop_capture = 0
        self.save_flag = 0
        self.manual_flag = 0
        self.split_size = split_size
        self.input_list = input_list

    def set_save_flag(self, flag=1):
        self.save_flag = flag

    def set_stop_capture(self, val=1):
        self.stop_capture = val

    def set_manual_flag(self, val=1):
        self.manual_flag = val
    
    def set_frame_update_time(self, val=5):
        self.frame_update_time = val

    def  insert_into_list(self, n_cam, n_frame, n_split, img):
        split_info = (n_frame, n_cam, n_split)
        split_value = {lm.IMAGE_KEY:img, lm.BINARY_KEY:None, lm.LOCALIZATION_KEY:None, lm.CLASSIFICATION_KEY:None}
        self.input_list.insert_image_split(split_info, split_value)

    def create_read_threads(self):
        self.read_threads = []
        if self.manual_flag:
            connected_cameras = list(map(str, range(1, 25)))
        else:
            connected_cameras = self.cameras.get_connected_cameras_by_id().keys()
        step = int(24 / self.n_threads)
        for i in range(self.n_threads):
            s = (i * step) + 1
            d = (i + 1) * step
            if self.start_cam <= s <= self.stop_cam or self.start_cam + 12 <= s <= self.stop_cam + 12:
                if any(str(camera_id) in connected_cameras for camera_id in list(range(s, d+1))):
                    self.read_threads.append(threading.Thread(target=self.read_image, args=(s, d)))

    def start_read_threads(self):
        for t in self.read_threads:
            t.start()

    def update_sheet(self, stop_cam, coil_dict):
        self.sheet_id = str(coil_dict['sheet_id'])
        self.nframe = [0] * 24
        self.stop_cam = stop_cam
        create_sheet_path(self.main_path, self.sheet_id)
        self.coil_dict = coil_dict
        print(coil_dict)

    def read_image(self, s, d):
        connected_cameras = self.cameras.get_connected_cameras_by_id()
        for camera_id in range(s, d + 1):
            if self.start_cam <= camera_id <= self.stop_cam or self.start_cam + 12 <= camera_id <= self.stop_cam + 12:
                if str(camera_id) in list(connected_cameras.keys()):
                    connected_cameras[str(camera_id)].start_grabbing()
                if self.stop_capture:
                    return
        while True:
            for camera_id in range(s, d + 1):
                if self.start_cam <= camera_id <= self.stop_cam or self.start_cam + 12 <= camera_id <= self.stop_cam + 12:
                    if str(camera_id) in list(connected_cameras.keys()):
                        ret, img = connected_cameras[str(camera_id)].getPictures()
                        if ret:
                            self.nframe[int(camera_id) - 1] += 1

                            # SPLIT
                            # t=time.time()
                            crops = ImageCrops_3d(img, self.split_size)

                            # INSERT INTO LIST
                            n_split = 1
                            for i in range(crops.shape[0]):
                                for j in range(crops.shape[1]):
                                    self.insert_into_list(camera_id, self.nframe[int(camera_id) - 1], n_split, crops[i, j])
                                    n_split+=1
                            # print((time.time() - t)*1000)

                            if self.save_flag:
                                if int(camera_id) <= 12:
                                    side = 'TOP'
                                    path = sheet_image_path(self.main_path, self.sheet_id, side, camera_id,
                                                            str(self.nframe[int(camera_id) - 1]),
                                                            self.image_format)

                                else:
                                    side = 'BOTTOM'
                                    path = sheet_image_path(self.main_path, self.sheet_id, side, str(camera_id - 12),
                                                            str(self.nframe[int(camera_id) - 1]),
                                                            self.image_format)
                                cv2.imwrite(path, img)

                    else:
                        if self.manual_flag:
                            self.nframe[int(camera_id) - 1] += 1
                            img = np.zeros((1200, 1920, 3), dtype=np.uint8)
                            img[:, :] = np.random.randint(0, 150)

                            # SPLIT
                            # t=time.time()
                            crops = ImageCrops_3d(img, self.split_size)
                            
                            # INSERT INTO LIST
                            n_split = 1
                            for i in range(crops.shape[0]):
                                for j in range(crops.shape[1]):
                                    self.insert_into_list(camera_id, self.nframe[int(camera_id) - 1], n_split, crops[i, j])
                                    n_split+=1
                            # print((time.time() - t)*1000)

                            if self.save_flag:
                                if int(camera_id) <= 12:
                                    side = 'TOP'
                                    path = sheet_image_path(self.main_path, self.sheet_id, side, camera_id,
                                                            str(self.nframe[int(camera_id) - 1]),
                                                            self.image_format)

                                else:
                                    side = 'BOTTOM'
                                    path = sheet_image_path(self.main_path, self.sheet_id, side, str(camera_id - 12),
                                                            str(self.nframe[int(camera_id) - 1]),
                                                            self.image_format)

                                cv2.imwrite(path, img)
                                
                            time.sleep(self.frame_update_time/1000)
                if self.stop_capture:
                    return
            if self.stop_capture:
                return

    def join_all(self):
        for t in self.read_threads:
            t.join()

    def start(self):
        print('start grab')
        self.set_stop_capture(val=0)

        self.create_read_threads()
        self.start_read_threads()

    def stop(self):
        self.set_stop_capture()
        self.join_all()
        print('stop grab')