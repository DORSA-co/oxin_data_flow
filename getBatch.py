import threading
import time

TIME_SLEEP = 0.001
global itr
itr=0

class get_batch():
    def __init__(self, n_get_batch_threads, input_list, gpu0_obj):
        self.n_threads = n_get_batch_threads
        self.input_list = input_list
        self.gpu0_obj = gpu0_obj
        self.stop_get_batch = 0
        self.force = False

    def set_stop_get_batch(self, val=1):
        self.stop_get_batch = val

    def set_force(self, val=True):
        self.force = val

    def create_threads(self):
        self.threads = []
        for i in range(self.n_threads):
            self.threads.append(threading.Thread(target=self.pop_batch))

    def start_threads(self):
        for t in self.threads:
            t.start()


    def pop_batch(self):
        global itr
        while True:
            # check if any of the gpus are empty
            if self.gpu0_obj.input_split_info is None:
                if self.input_list.len_list>0:
                    print(self.input_list.len_list)

                # get batch from list
                if not self.force:
                    split_info, split_values, split_images = self.input_list.get_batch(force=False)
                else:
                    split_info, split_values, split_images = self.input_list.get_batch(force=True)
                
                # set to gpu input if not None
                # if split_info is not None and split_values is not None:
                #     while True:
                #         res = self.gpu0_obj.set_input_batch(input_split_infoes_batch=split_info,
                #                                             input_split_values_batch=split_values,
                #                                             input_split_image_batch=split_images)

                #         if res:
                #             break
                #         # break
                #         itr+=1
                #         time.sleep(TIME_SLEEP)
                    
                # stop by stoping get batch
                if self.stop_get_batch:
                    return
            
            else:
                itr+=1
            
            # print('itr', itr)
            time.sleep(TIME_SLEEP)


    def join_all(self):
        for t in self.threads:
            t.join()

    def start(self):
        print('start get batch')
        self.set_stop_get_batch(val=0)

        self.create_threads()
        self.start_threads()

    def stop(self):
        self.set_stop_get_batch()
        self.join_all()
        print('stop get batch')

    
