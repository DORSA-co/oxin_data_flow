import threading
import time

TIME_SLEEP = 0.001
BATCH_SIZE = 128

class gpu_threads():
    def __init__(self, n_gpu_threads, gpu0_obj):
        self.n_threads = n_gpu_threads
        self.gpu0_obj = gpu0_obj
        self.stop_gpu = 0

    def set_stop_gpu(self, val=1):
        self.stop_gpu = val

    def create_threads(self):
        self.threads = []
        for i in range(self.n_threads):
            self.threads.append(threading.Thread(target=self.gpu_run_inference))

    def start_threads(self):
        for t in self.threads:
            t.start()

    def gpu_run_inference(self):
        while True:
            start = time.time()

            res = self.gpu0_obj.run_inference_binary()


            # stop by stoping get batch
            if self.stop_gpu:
                return

            time.sleep(TIME_SLEEP)

            if res:
                print('FPS: %s' % str(BATCH_SIZE/(time.time()-start)))


    def join_all(self):
        for t in self.threads:
            t.join()

    def start(self):
        print('start gpu')
        self.set_stop_gpu(val=0)

        self.create_threads()
        self.start_threads()

    def stop(self):
        self.set_stop_gpu()
        self.join_all()
        print('stop gpu')

    
