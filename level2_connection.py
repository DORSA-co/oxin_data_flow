
import numpy as np
RANDOM=True
class connection_level2():

    def init(self,ui_obj):
        self.ui_obj=ui_obj
        pass

    def ret_sheet_details(self):
        ret,details=self.connect()
        return ret,details
    
    
    def calculate_camera_and_projector(self):

        #calculate number of camera and projector
        ret,details=self.ret_sheet_details()
        if ret:
            cameras=int(details['width']/14 )     # this should be change 
            # projector =int(details['width'] /14)
            cameras=24
            projector=4
            return cameras,projector,details
 
        else:
            return 0,0,None
 


    def get_number_camera_projector(self):

        cameras,projectors,_=self.calculate_camera_and_projector()
        return cameras,projectors

    def get_full_info(self):
        cameras,projectors,details=self.calculate_camera_and_projector()
        return cameras,projectors,details


    def connect(self):

        # Should add connection function here and get details
        try:
            details={}
            if RANDOM:
                details.update({"sheet_id": np.random.randint(10,50)})
                details.update({"pdl_number": np.random.randint(100,1000)})
                details.update({"heat_number": np.random.randint(100,1000)})
                details.update({"ps_number": np.random.randint(100,1000)})
                details.update({"width": np.random.randint(5,100)})
                details.update({"lenght": np.random.randint(1000,10000)})
                details.update({"thickness": np.random.randint(10,50)})
                self.details=details
                return True,details
        except:
            return False,None



if __name__=='__main__':

    conn=connection_level2()
    cameras,projectors,details = conn.get_full_info()
    print(cameras,projectors,details )
        