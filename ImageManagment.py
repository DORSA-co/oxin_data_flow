import threading
import time
import logging
import level2_connection
import camera_connection
import list_management_dict as lm
import dataGrabber
import getBatch
import gpuThreads
import gpu
# import desplit_images


TIME_SLEEP = 0.001 # for thread s to prevent code to get slow
BATCH_SIZE = 128
INPUT_SHAPE = (224,224,3)
SPLIT_SIZE = (224, 224, 3)
DETECT_TIME = 8
NODETECT_TIME = 20


class Data_Flow_Handler():
    def __init__(self, n_grab_threads, n_get_batch_threads, n_gpu_threads, cameras, main_path, engine_path="model/binary_model_0.plan"):
        self.start_detect_flag = False

        # set number of threads
        self.n_grab_threads = n_grab_threads
        self.n_get_batch_threads = n_get_batch_threads
        self.n_gpu_threads = n_gpu_threads

        # init lists
        self.input_list = lm.List_Management(batch_size=BATCH_SIZE, image_shape=SPLIT_SIZE, sensitive_safety=True)
        self.perfect_list = lm.List_Management(batch_size=BATCH_SIZE, image_shape=SPLIT_SIZE, sensitive_safety=True)
        self.defect_list = lm.List_Management(batch_size=BATCH_SIZE, image_shape=SPLIT_SIZE, sensitive_safety=True)

        # init modules
        ## data grabber
        self.data_grabber = dataGrabber.data_grabber(cameras, self.n_grab_threads, main_path, SPLIT_SIZE, self.input_list)
        self.set_grabber_manual_flag()
        self.set_grabber_save_flag()
        self.set_grabber_frame_update_time(val=80)

        ## gpu 
        self.gpu0 = gpu.GPU_Handler()
        self.gpu0.assign_parameters(gpu_idx=0, thread_idx=0, engine_path=engine_path,
                                batch_size=BATCH_SIZE, input_width=INPUT_SHAPE[0], input_height=INPUT_SHAPE[1], n_channel=INPUT_SHAPE[2],
                                n_classes=1000,
                                defect_list_obj=self.defect_list, perfect_list_obj=self.perfect_list)

        ## get batch
        self.get_batch = getBatch.get_batch(self.n_get_batch_threads, self.input_list, self.gpu0)

        ## gpu threads
        self.gpu_threads = gpuThreads.gpu_threads(self.n_gpu_threads, self.gpu0)


    def set_grabber_save_flag(self, flag=1):
        self.data_grabber.set_save_flag(flag)

    def set_grabber_stop_capture(self, val=1):
        self.data_grabber.set_stop_capture(val)

    def set_grabber_manual_flag(self, val=1):
        self.data_grabber.set_manual_flag(val)
    
    def set_grabber_frame_update_time(self, val=5):
        self.data_grabber.set_frame_update_time(val)

    def grabber_update_sheet(self, stop_cam, coil_dict):
        self.data_grabber.update_sheet(stop_cam, coil_dict)

    def set_get_batch_force(self, val=True):
        self.get_batch.set_force(val)

    def set_get_batch_stop(self, val=1):
        self.get_batch.set_stop_get_batch(val)

    def set_gpu_threads_stop(self, val=1):
        self.gpu_threads.set_stop_gpu(val)

    def start(self):
        if not self.start_detect_flag:
            self.data_grabber.start()
            self.get_batch.start()
            self.gpu_threads.start()
            self.start_detect_flag = True
        else:
           self.data_grabber.start() 

    def stop(self):
        self.data_grabber.stop()
        self.get_batch.set_force()

def detect_sensor_simulator():
    # print('here')
    cameras = camera_connection.connect_manage_cameras()
    # connect_camera(cameras)
    data_flow = Data_Flow_Handler(n_grab_threads=12, n_get_batch_threads=2, n_gpu_threads=1, cameras=cameras, main_path='oxin_image_grabber')
    
    l2_connection = level2_connection.connection_level2()
    detect = False
    t1 = time.time()
    while True:
        t2 = time.time()

        # detection ended
        if detect and t2-t1 >= DETECT_TIME:
            detect = False
            t1 = time.time()

            ## STOP
            data_flow.stop()
        
        # no detection ended
        elif not detect and t2-t1 >= NODETECT_TIME:
            detect = True
            t1 = time.time()

            ## START
            (n_camera, projectors, details) = l2_connection.get_full_info()
            data_flow.grabber_update_sheet(stop_cam=n_camera, coil_dict=details)
            data_flow.start()

        time.sleep(TIME_SLEEP)

def connect_camera(cameras):
        """
        This function is used to connect selected cameras
        """
        selected_cameras = [True]*24

        if not any(selected_cameras):
            print('no camera selected')
            return

        for i in range(len(selected_cameras)):
            if selected_cameras[i]:
                cam_num = i + 1
                cam_parms = get_camera_config(cam_num)
                ret = cameras.add_camera(str(cam_num), cam_parms)

                if ret == "True":
                    print('camera {} connected successfully'.format(cam_num))

                else:
                    if ret == "Camera Not Connected":
                        print('Serial number of camera {} is not availabla'.format(cam_num))
                    else:
                        print('Camera {} controlled by another application Or config error'.format(cam_num))

def get_camera_config(cam_num):
    camera_parms = [
        {'id': 1, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '192.168.1.100', 'rotation_value': 3.7, 'shifth_value': -39, 'shiftw_value': 2, 'serial_number': '24350287', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 2, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '192.168.1.2', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '24350353', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 3, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '24350360', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 4, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '24350361', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 5, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 2.0, 'shifth_value': 21, 'shiftw_value': 50, 'serial_number': '24350366', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 6, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '24350367', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 7, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 8, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 9, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 10, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 11, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 12, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 13, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 14, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 15, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 16, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 17, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 18, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 19, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 20, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 21, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 22, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 23, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0},
        {'id': 24, 'gain_top': 0, 'gain_bottom': 360, 'gain_value': 0, 'expo_top': 35, 'expo_bottom': 10000000, 'expo_value': 500, 'width': 1920, 'height': 1200, 'offsetx_top': 16, 'offsetx_bottom': 0, 'offsetx_value': 0, 'offsety_top': 16, 'offsety_bottom': 0, 'offsety_value': 0, 'interpacket_delay': 10, 'packet_size': 10, 'trigger_mode': 0, 'max_buffer': 10, 'transmission_delay': 10, 'ip_address': '', 'rotation_value': 0.0, 'shifth_value': 0, 'shiftw_value': 0, 'serial_number': '0', 'pxvalue_a': 0.0, 'pxvalue_b': 0.0, 'pxvalue_c': 0.0}
    ]
    return camera_parms[cam_num-1]



if __name__ == '__main__':
    detect_sensor_simulator()
