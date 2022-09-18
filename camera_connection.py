"""
########################################
---------------------------------------

Made with Malek & Milad

Features:

    ● Create Unlimite Object of Cameras and Live Preview By serial number
    ● Set Bandwitdh Of each Cameras
    ● Set gain,exposure,width,height,offet_x,offset_y
    ● Get tempreture of Cmeras
    ● Set Trigger Mode on
    ● There are Some diffrents between ace2(pro) and ace

---------------------------------------
########################################
"""

from pickle import FALSE
from pypylon import pylon
import cv2
import time
import numpy as np
import sqlite3
import threading

from pypylon import genicam

DEBUG = False

# import database_utils

show_eror = False

if show_eror:

    from eror_window import UI_eror_window


class Collector():

    def __init__(self, serial_number,gain = 0 , exposure = 70000, max_buffer = 20, trigger=True, delay_packet=100, packet_size=1500 ,
                frame_transmission_delay=0 ,width=1000,height=1000,offet_x=0,offset_y=0, manual=False, list_devices_mode=False, trigger_source='Software'):
        """Initializes the Collector
        Args:
            gain (int, optional): The gain of images. Defaults to 0.
            exposure (float, optional): The exposure of the images. Defaults to 3000.
            max_buffer (int, optional): Image buffer for cameras. Defaults to 5.
        """
        self.gain = gain
        self.exposure = exposure
        self.max_buffer = max_buffer
        self.cont_eror=0
        self.serial_number = serial_number
        self.trigger = trigger
        self.trigger_source = trigger_source
        self.dp = delay_packet
        self.ps=packet_size
        self.ftd=frame_transmission_delay
        self.width=width
        self.height=height
        self.offset_x=offet_x
        self.offset_y=offset_y
        self.manual=manual
        self.list_devices_mode=list_devices_mode
        self.exitCode=0

        if show_eror:
            self.window_eror = UI_eror_window()

        self.__tl_factory = pylon.TlFactory.GetInstance()
        devices = []


        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


        for device in self.__tl_factory.EnumerateDevices():
            if (device.GetDeviceClass() == 'BaslerGigE'):                
                devices.append(device)

        # assert len(devices) > 0 , 'No Camera is Connected!
        if self.list_devices_mode:
            self.cameras = list()

            for device in devices:
                camera = pylon.InstantCamera(self.__tl_factory.CreateDevice(device))
                self.cameras.append(camera)
        
        else:
            for device in devices:
                camera = pylon.InstantCamera(self.__tl_factory.CreateDevice(device))
                # print(camera.GetDeviceInfo().GetSerialNumber())
                if camera.GetDeviceInfo().GetSerialNumber() == self.serial_number:
                    self.camera = camera
                
                    break

        #assert len(devices) > 0 , 'No Camera is Connected!'
        


    def eror_window(self,msg,level):
        self.window_eror = UI_eror_window()
       # self.ui2= UI_eror_window()
        self.window_eror.show()
        self.window_eror.set_text(msg,level)


    def tempreture(self):
        device_info = self.camera.GetDeviceInfo()
        model=str(device_info.GetModelName())
        model=model[-3:]
        if model=='PRO':
            # print(self.camera.DeviceTemperature.GetValue())
            return self.camera.DeviceTemperature.GetValue()
        else :
            # print('temp',self.camera.TemperatureAbs.GetValue())
            return self.camera.TemperatureAbs.GetValue()


    def start_grabbing(self):

        device_info = self.camera.GetDeviceInfo()
        model=str(device_info.GetModelName())
        model=model[-3:]
        # print(model[-3:])


        try:
            # print(self.camera.IsOpen())
            # print(device_info.GetSerialNumber())

            self.camera.Open()
            
            if self.manual:

                
                if model=='PRO':
                    # print('yes pro')
                    # print(self.camera.DeviceTemperature.GetValue())
                    self.camera.ExposureTime.SetValue(self.exposure)

                    self.camera.Gain.SetValue(self.gain)
                    
                    # self.camera.GevSCPSPacketSize.SetValue(int(self.ps)+1000)
                    # self.camera.Close()
                    # self.camera.Open()
                    self.camera.GevSCPSPacketSize.SetValue(int(self.ps))
                    self.camera.Close()
                    self.camera.Open()
                                                  
                    self.camera.GevSCPD.SetValue(self.dp)
                    self.camera.Close()
                    self.camera.Open()                   
                    self.camera.GevSCFTD.SetValue(self.ftd)
                    self.camera.Close()
                    self.camera.Open()




                    self.camera.Width.SetValue(self.width)
                    self.camera.Height.SetValue(self.height)

                    self.camera.OffsetX.SetValue(self.offset_x)
                    self.camera.OffsetY.SetValue(self.offset_y)
                    



                

                else:
                    


                    self.camera.ExposureTimeAbs.SetValue(self.exposure)
                    self.camera.GainRaw.SetValue(self.gain)

                    self.camera.GevSCPSPacketSize.SetValue(int(self.ps)+1000)
                    self.camera.Close()
                    self.camera.Open()
                                
                    self.camera.GevSCPD.SetValue(self.dp)
                    self.camera.Close()
                    self.camera.Open()                   
                    self.camera.GevSCFTD.SetValue(self.ftd)
                    self.camera.Close()
                    self.camera.Open()

                    self.camera.GevSCPSPacketSize.SetValue(int(self.ps))
                    self.camera.Close()
                    self.camera.Open()
                    self.camera.Width.SetValue(self.width)
                    self.camera.Height.SetValue(self.height)

                    self.camera.OffsetX.SetValue(self.offset_x)
                    self.camera.OffsetY.SetValue(self.offset_y)
                    


            self.camera.Close()

            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 

            self.camera.Open()

            if self.trigger:
                self.camera.TriggerSelector.SetValue('FrameStart')
                self.camera.TriggerMode.SetValue('On')
                self.camera.TriggerSource.SetValue(self.trigger_source)
                # print('triggeron on %s' % self.trigger_source)
            else:
                pass
                # self.camera.TriggerMode.SetValue('Off')
                # print('triggeroff')

            # if self.manual:
            #     self.camera.ExposureTimeAbs.SetValue(20000)


            #     # self.camera.Width.SetValue(600)
            #     print(self.camera.Width.GetValue())
            #     self.camera.Width.SetValue(600)
            #     # int64_t = self.camera.PayloadSize.GetValue()
            #     # self.camera.GevStreamChannelSelectorCamera.GevStreamChannelSelector.SetValue( 'GevStreamChannelSelector_StreamChannel0 ')
            #     # self.camera.GevSCPSPacketSize.SetValue(1500)
                             
            #     self.camera.GevSCPD.SetValue(self.dp)
                
            #     self.camera.GevSCFTD.SetValue(self.ftd)
            self.exitCode=0

            return True, 'start grabbing ok'
            
        except genicam.GenericException as e:
            # Error handling
            
            message = self.start_grabbing_error_handling(error=e)
            #print(e)  
        
            self.stop_grabbing()
            #print("An exception occurred.", e.GetDescription())
            self.exitCode = 1
            # self.eror_window('Check The Number of cameras',3)

            
            return False, message

    
    def start_grabbing_error_handling(self, error):
        message = ''
        # camera in use
        if 'The device is controlled by another application' in str(error):
            message = 'Camera is controlled by another application'

        # expossure invalid
        elif "OutOfRangeException thrown in node 'ExposureTimeAbs' while calling 'ExposureTimeAbs.SetValue()" in str(error):
            # min
            if 'greater than or equal' in str(error):
                message = 'Exposure value is too small'
            elif 'must be smaller than or equal' in str(error):
                message = 'Exposure value is too large'
            else:
                message = 'Exposure value invalid'

        # gain invalid
        elif "OutOfRangeException thrown in node 'GainRaw' while calling 'GainRaw.SetValue()" in str(error):
            if 'must be equal or greater than' in str(error):
                message = 'Gain value is too small'
            elif 'must be equal or smaller than' in str(error):
                message = 'Gain value is too large'
            else:
                message = 'Gain value invalid'

        # packetsize invalid
        elif "OutOfRangeException thrown in node 'GevSCPSPacketSize' while calling 'GevSCPSPacketSize.SetValue()" in str(error):
            message = 'Packet-size value invalid'
        
        # transmission delay
        elif "OutOfRangeException thrown in node 'GevSCFTD' while calling 'GevSCFTD.SetValue()" in str(error):
            if 'must be equal or greater than' in str(error):
                message = 'Transmision delay is too small'
            elif 'must be equal or smaller than' in str(error):
                message = 'Transmision delay is too large'
            else:
                message = 'Transmision delay value invalid'

        # height delay
        elif "OutOfRangeException thrown in node 'Height' while calling 'Height.SetValue()" in str(error):
            if 'must be equal or greater than' in str(error):
                message = 'Height value is too small'
            elif 'must be equal or smaller than' in str(error):
                message = 'Height value is too large'
            else:
                message = 'Height value invalid'

        # width delay
        elif "OutOfRangeException thrown in node 'Width' while calling 'Width.SetValue()" in str(error):
            if 'must be equal or greater than' in str(error):
                message = 'Width value is too small'
            elif 'must be equal or smaller than' in str(error):
                message = 'Width value is too large'
            else:
                message = 'Width value invalid'

        # offsetx delay
        elif "OutOfRangeException thrown in node 'OffsetX' while calling 'OffsetX.SetValue()" in str(error):
            if 'must be equal or greater than' in str(error):
                message = 'Offsetx value is too small'
            elif 'must be equal or smaller than' in str(error):
                message = 'Offsetx value is too large'
            else:
                message = 'Offsetx value invalid'

        # offsety delay
        elif "OutOfRangeException thrown in node 'OffsetY' while calling 'OffsetY.SetValue()" in str(error):
            if 'must be equal or greater than' in str(error):
                message = 'Offsety value is too small'
            elif 'must be equal or smaller than' in str(error):
                message = 'Offsety value is too large'
            else:
                message = 'Offsety value invalid'
        

        else:
            message = str(error)

        return message



    def stop_grabbing(self):
        self.camera.Close()

            
        
    def listDevices(self):
        """Lists the available devices
        """
        for i ,  camera in enumerate(self.cameras):
            device_info = camera.GetDeviceInfo()
            # print(
            #     "Camera #%d %s @ %s (%s) @ %s" % (
            #     i,
            #     device_info.GetModelName(),
            #     device_info.GetIpAddress(),
            #     device_info.GetMacAddress(),
            #     device_info.GetSerialNumber(),
            #     )
            
            # )
            # print(device_info)


    def serialnumber(self):
        serial_list=[]
        for i ,  camera in enumerate(self.cameras):
            device_info = camera.GetDeviceInfo()
            serial_list.append(device_info.GetSerialNumber())
        return serial_list         




    def trigg_exec(self,):
        
        if self.trigger:
            self.camera.TriggerSoftware()
            #print(self.camera.GetQueuedBufferCount(), 'T'*100)
            while self.camera.GetQueuedBufferCount() >=10:
                pass
            #print(self.camera.GetQueuedBufferCount(), 'T'*100)


    def getPictures(self, time_out = 50):
        Flag=True
        try:

            
            # if DEBUG:
            #     print('TRIGE Done')
            # print('444444444444', self.camera.IsGrabbing())
            if self.camera.IsGrabbing():
                # if DEBUG:
                    # print('Is grabbing')
                    
                    # if self.camera.GetQueuedBufferCount() == 10:
                        # print('ERRRRRRRRRRRRRRRRRRRRRRRRRROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOORRRRRRRRRRRRRRRRRRRRRRRRR')
                grabResult = self.camera.RetrieveResult(time_out, pylon.TimeoutHandling_ThrowException)

                # print('grab',grabResult)
                

                # print(self.camera.GetQueuedBufferCount(), 'f'*100)
                # if DEBUG:
                #     # print('RetrieveResult')

                #     if self.camera.GetQueuedBufferCount() == 10:
                #         print('ERRRRRRRRRRRRRRRRRRRRRRRRRROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOORRRRRRRRRRRRRRRRRRRRRRRRR')
                if grabResult.GrabSucceeded():
                    
                    # if DEBUG:
                    #     print('Grab Succed')

                    image = self.converter.Convert(grabResult)
                    img=image.Array

                else:
                    img=np.zeros([1200,1920,3],dtype=np.uint8)
                    self.cont_eror+=1
                    # print('eror',self.cont_eror)
                    # print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
                    Flag=False

            else:
                    # print('erpr')
                    img=np.zeros([1200,1920,3],dtype=np.uint8)
                    Flag=False

        except:
            #print('Time out')
            img=np.zeros([1200,1920,3],dtype=np.uint8)
            Flag=False

        # cv2.imshow("img1", cv2.resize(img, None, fx=0.5, fy=0.5))
        # cv2.waitKey(50)
        if Flag:
            #print('yes')
            return True, img
        else:
            #print('no')
            return False, np.zeros([1200,1920,3],dtype=np.uint8)



    def get_cam(self,i):
        return self.camera
    


def get_threading(cameras):
    def thread_func():
        for cam in cameras:
            cam.trigg_exec()
        for cam in cameras:
            img = cam.getPictures()
            cv2.imshow("img", cv2.resize(img, None, fx=0.5, fy=0.5))
            cv2.waitKey(10)

        t = threading.Timer(0.330, thread_func)
        t.start()

    return thread_func


class connect_manage_cameras:
    def __init__(self):

        self.id_list = []
        # self.db=database_utils.dataBaseUtils()
        self.cam_objs = {}
        self.cam_ids = {}
        self.get_all_devices()

    # def add_camera(self,id,cam_parms):

    #     if id not in self.id_list:

    #         self.id_list.append(id)

    #         self.create_new_connection(id,cam_parms)

    #     else :

    #         # check connection
    #         print('This camera is Connected')

    def add_camera(self, id, cam_parms):

        # create_connection_based on data base
        # cam_parms=self.get_camera_config(id)

        # _, available_serials = self.get_all_devices()
        available_serials = self.list_available_serial
        # print("cam_adwdw", cam_parms)

        # print("asdw", available_serials, str(cam_parms["serial_number"]))

        if str(cam_parms["serial_number"]) in available_serials:

            # print(cam_parms['serial_number'])

            # try:

            collector = Collector(
                str(cam_parms["serial_number"]),
                exposure=cam_parms["expo_value"],
                gain=cam_parms["gain_value"],
                trigger=False,
                delay_packet=cam_parms["interpacket_delay"],
                packet_size=cam_parms["packet_size"],
                frame_transmission_delay=cam_parms["transmission_delay"],
                height=cam_parms["height"],
                width=cam_parms["width"],
                offet_x=cam_parms["offsetx_value"],
                offset_y=cam_parms["offsety_value"],
                manual=False,
            )

            # print(collector)

            ret = collector.start_grabbing()

            self.cam_objs[cam_parms["serial_number"]] = collector
            self.cam_ids[id] = collector
            # print("grab_cameras", self.cam_objs)

            # print("ret grab", ret)

            collector.stop_grabbing()

            if ret[0]:

                return "True"

            else:
                return ret

        # except:

        #     # print(ret)

        #     return "False"

        else:

            return "Camera Not Connected"

    def get_camera_config(self, id):

        self.db = database_utils.dataBaseUtils()

        cam_parms = self.db.load_cam_params(id)

        # print(cam_parms)

        return cam_parms

    def get_all_devices(self):

        self.__tl_factory = pylon.TlFactory.GetInstance()
        devices = []

        for device in self.__tl_factory.EnumerateDevices():
            if device.GetDeviceClass() == "BaslerGigE":
                devices.append(device)

        # assert len(devices) > 0 , 'No Camera is Connected!'

        self.all_available_cameras = list()

        for device in devices:
            try:
                camera = pylon.InstantCamera(self.__tl_factory.CreateDevice(device))
                self.all_available_cameras.append(camera)
            except:
                pass

        # return

        serial_list = []
        for i, camera in enumerate(self.all_available_cameras):
            device_info = camera.GetDeviceInfo()
            serial_list.append(device_info.GetSerialNumber())
        self.list_available_serial = serial_list
        return self.all_available_cameras, serial_list

    def check_my_cameras_connected(self):

        serial_list = self.get_all_devices()

        if len(serial_list) != len(self.cam_ids.keys()):
            self.disconected_cams = []
            for cam in serial_list:
                if cam not in list(self.cam_objs.keys()):
                    self.disconected_cams.append(self.cam_objs)
        return self.disconected_cams

    def disconnect_camera(self, sn, id):

        try:
            if sn in self.cam_objs:
                cam_obj = self.cam_objs[sn]
                # print("cam_obj", cam_obj)
                cam_obj.stop_grabbing()
                self.cam_objs.pop(sn)
                self.cam_ids.pop(id)
                return "True"
            else:
                return "no_connection"

        except:
            raise False
            return False

    def get_connected_cameras(self):

        return self.cam_objs

    def get_connected_cameras_by_id(self):
        # self.cam_ids = {'1': None, '2':None, '13':None, '14':None}
        return self.cam_ids


def get_all_devices():

    tl_factory = pylon.TlFactory.GetInstance()

    cam = None
    # tlf = pylon.TlFactory.GetInstance()

    # for tl in tlf.EnumerateTls():
    #     print(tl.GetDeviceClass(), tl.GetFileName(), tl.GetFullName())
    for dev_info in tl_factory.EnumerateDevices():
        if dev_info.GetDeviceClass() == "BaslerGigE":
            # print("using %s @ %s" % (dev_info.GetModelName(), dev_info.GetIpAddress()))
            cam = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
            # break
    return False


if __name__ == "__main__":

    # cam_obj=connect_manage_cameras()

    # parms=cam_obj.get_camera_config('4')

    # cam_obj.add_camera('4',parms)

    # cameras = {}
    # # for sn in ['40150887']:
    #     # collector = Collector( sn,exposure=3000 , gain=30, trigger=False, delay_packet=170000)
    collector = Collector(
        "24350367",
        exposure=3000,
        gain=10,
        trigger=False,
        delay_packet=81062,
        packet_size=9000,
        frame_transmission_delay=18036,
        height=1000,
        width=1000,
        offet_x=16,
        offset_y=4,
        manual=False,
    )

    # x=collector.get_cam()

    # collector.start_grabbing()
    # collector.start_grabbing()
    cameras = collector

    cameras.start_grabbing()
    cameras.getPictures()
    # print(cameras.)

    while True:

        #     # for cam in cameras:
        #     #         cam.trigg_exec()

        #     # for cam in cameras:
        #     #print(cam.camera.GetQueuedBufferCount())
        img = cameras.getPictures()
        img=img[1]
        # print(img.shape)
        # print(cam.camera.GetQueuedBufferCount())
        cv2.imshow("img1", cv2.resize(img, None, fx=0.5, fy=0.5))
        img=np.uint8(img)
        # cv2.imshow('img',img)
        cv2.waitKey(50)
        # img = cameras[1].getPictures()
        # #print(cam.camera.GetQueuedBufferCount())
        # cv2.imshow('img2', cv2.resize( img, None, fx=0.5, fy=0.5 ))
        # cv2.waitKey(50)
        # img = cameras[2].getPictures()
        # #print(cam.camera.GetQueuedBufferCount())
        # cv2.imshow('img3', cv2.resize( img, None, fx=0.5, fy=0.5 ))
        # cv2.waitKey(50)
        # img = cameras[3].getPictures()
        # #print(cam.camera.GetQueuedBufferCount())
        # cv2.imshow('img4', cv2.resize( img, None, fx=0.5, fy=0.5 ))
        # cv2.waitKey(50)

        # time.sleep(0.330)
        # while cam.camera.GetQueuedBufferCount()!=10:
        #     pass
        # print(cam.camera.GetQueuedBufferCount(), 'f'*100)
        # print('-'*100)
    # func = get_threading(cameras)
    # func()
