# from tensorflow import keras
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
#from zmq import device 
import engine as eng
import threading
import time

#TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#trt_runtime = trt.Runtime(TRT_LOGGER)


class Inference:

   def __init__(self, engine_path,  cuda_idx=0):

      #cuda_idx = 0

      self.cuda_idx = cuda_idx
      print('Selected Device is {}'.format(cuda.Device( self.cuda_idx ).name()))
      self.cfx = cuda.Device( self.cuda_idx ).make_context()
      
      self.stream = cuda.Stream()


      TRT_LOGGER = trt.Logger(trt.Logger.INFO)
      trt.init_libnvinfer_plugins(TRT_LOGGER, '' )
      trt_runtime = trt.Runtime(TRT_LOGGER)

      
      self.engine = eng.load_engine(trt_runtime, engine_path)    
      self.context = self.engine.create_execution_context()

      
      
   #_________________________________________________________________________________________________________________________________________
   #
   #_________________________________________________________________________________________________________________________________________
   def allocate_buffers(self, batch_size, data_type):

      """
      This is the function to allocate buffers for input and output in the device
      Args:
         engine : The path to the TensorRT engine. 
         batch_size : The batch size for execution time.
         data_type: The type of the data for input and output, for example trt.float32. 
      
      Output:
         h_input_1: Input in the host.
         d_input_1: Input in the device. 
         h_output_1: Output in the host. 
         d_output_1: Output in the device. 
         stream: CUDA stream.

      """

      host_inputs  = []
      cuda_inputs  = []
      host_outputs = []
      cuda_outputs = []
      bindings = []
      #bindings are inputs and outputs
      for i in range(len(self.engine)):
         size = trt.volume( self.engine.get_binding_shape(i)[:] )* self.engine.max_batch_size
         host_mem = cuda.pagelocked_empty(size, dtype=trt.nptype(data_type) )
         cuda_mem = cuda.mem_alloc(host_mem.nbytes)

         bindings.append(int(cuda_mem))
         if self.engine.binding_is_input( self.engine[i] ):
               host_inputs.append(host_mem)
               cuda_inputs.append(cuda_mem)
         else:
               host_outputs.append(host_mem)
               cuda_outputs.append(cuda_mem)


      self.host_inputs = host_inputs
      self.cuda_inputs = cuda_inputs
      self.host_outputs = host_outputs
      self.cuda_outputs = cuda_outputs
      self.bindings = bindings

      # self.h_input_1 = host_inputs[0]
      # self.h_output = host_outputs[0]
      # self.d_input_1 = cuda_inputs[0]
      # self.d_output = cuda_outputs[0]

      # # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
      # self.h_input_1 = cuda.pagelocked_empty(  trt.volume(self.engine.get_binding_shape(0)[:]), dtype=trt.nptype(data_type))
      # self.h_output = cuda.pagelocked_empty(  trt.volume(self.engine.get_binding_shape(1)[:]), dtype=trt.nptype(data_type))
      # # Allocate device memory for inputs and outputs.
      # self.d_input_1 = cuda.mem_alloc(self.h_input_1.nbytes)

      # self.d_output = cuda.mem_alloc(self.h_output.nbytes)




      return self.host_inputs, self.cuda_inputs, self.host_outputs, self.cuda_outputs, self.bindings 

   def load_images_to_buffer(self, pics, pagelocked_buffer):
      preprocessed = np.asarray(pics).ravel()
      np.copyto(pagelocked_buffer[0], preprocessed) 

   #_________________________________________________________________________________________________________________________________________
   #
   #_________________________________________________________________________________________________________________________________________
   def do_inference(self, pics_1, batch_size, height, width, reshape=None):
      """
      This is the function to run the inference
      Args:
         engine : Path to the TensorRT engine 
         pics_1 : Input images to the model.  
         h_input_1: Input in the host         
         d_input_1: Input in the device 
         h_output_1: Output in the host 
         d_output_1: Output in the device 
         stream: CUDA stream
         batch_size : Batch size for execution time
         height: Height of the output image
         width: Width of the output image
      
      Output:
         The list of output images

      """
      #threading.Thread.__init__(self)
      #self.cfx.push()
      self.cfx.push()
      #-----------------------------------
      stream  = self.stream
      context = self.context
      engine  = self.engine

      host_inputs = self.host_inputs
      cuda_inputs = self.cuda_inputs
      host_outputs = self.host_outputs
      cuda_outputs = self.cuda_outputs
      bindings = self.bindings
      #------------------------------------
      self.load_images_to_buffer(pics_1, host_inputs)
      
      #------------------------------------
      start = time.time()
      cuda.memcpy_htod_async( cuda_inputs[0], host_inputs[0], stream ) #send input host to cuda
      
      #print('1:', time.time() - start)
      context.execute_async( bindings=bindings, stream_handle=stream.handle)
      #print('2:', time.time() - start)
      cuda.memcpy_dtoh_async( host_outputs[0], cuda_outputs[0], stream)#get output cuda to host
      #print('3:', time.time() - start)
      #------------------------------------
      stream.synchronize()
      #print(host_outputs[0])
      while not stream.is_done():
         pass
      #print(stream.is_done())
     # print(host_outputs[0])
      #print('---------------------------------------------------------------------------')
      #print('4:', time.time() - start)
      #------------------------------------
      out = 0
      if reshape is not None:
         out = host_outputs[0].reshape((batch_size,-1, height, width))
      else:
         out = host_outputs[0]

      # try:
      #    if out == 0:
      #       print('======================================================== out zero')
            
      # except:
      #    if np.sum(out) == 0:
      #       print('====================================================================================================================== out sum zero')
      #    print('======================================================== out', out)
      #    pass

      #self.cfx.pop()

      self.cfx.pop()
      self.resault = out
      return out

      # with self.engine.create_execution_context() as context:
      #    # Transfer input data to the GPU.
      #    cuda.memcpy_htod_async(self.d_input_1, self.h_input_1, self.stream)

      #    # Run inference.

      #    context.profiler = trt.Profiler()
      #    context.execute(batch_size=1, bindings=[int(self.d_input_1), int(self.d_output)])

      #    # Transfer predictions back from the GPU.
      #    cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
      #    # Synchronize the stream
      #    self.stream.synchronize()
      #    # Return the host output.

      #    #self.cfx.pop()
      #    if reshape is not None:
      #       out = self.h_output.reshape((batch_size,-1, height, width))
      #       return out 
      #    else:
      #       return self.h_output

   def test(self):
      stream  = self.stream
      context = self.context
      engine  = self.engine

      host_inputs = self.host_inputs
      cuda_inputs = self.cuda_inputs
      host_outputs = self.host_outputs
      cuda_outputs = self.cuda_outputs
      bindings = self.bindings

      cuda.memcpy_dtoh_async( host_outputs[0], cuda_outputs[0], stream)#get output cuda to host

      out = 0
      reshape = None

      # if reshape is not None:
      #    out = host_outputs[0].reshape((batch_size,-1, height, width))
      # else:
      out = host_outputs[0]

      try:
         if out == 0:
            print('======================================================== out zero')
            
      except:
         if np.sum(out) == 0:
            print('====================================================================================================================== out sum zero')
         print('======================================================== out', out)
         pass

      #self.cfx.pop()
      self.resault = out
      return out
   
   #_________________________________________________________________________________________________________________________________________
   #
   #_________________________________________________________________________________________________________________________________________
   def destory(self):
        self.cfx.pop()


# x=Inference('k',)


