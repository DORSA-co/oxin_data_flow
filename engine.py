try:
    import tensorrt as trt
except:
    pass



def build_engine(onnx_path, shape = [1,224,224,3], precision='fp16'):

    """
    This is the function to create the TensorRT engine
    Args:
    onnx_path : Path to onnx_file. 
    shape : Shape of the input of the ONNX file. 
    """
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    trt_runtime = trt.Runtime(TRT_LOGGER)

    #os.environ['CUDA_DEVICE'] = '1'
    
    # builder = trt.Builder(TRT_LOGGER)
    # network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # parser = trt.OnnxParser(network, TRT_LOGGER)
    # #-----------------------------------------------------
    # success = parser.parse_from_file(onnx_path)
    # for idx in range(parser.num_errors):
    #     print(parser.get_error(idx))

    # if not success:
    #     assert False, 'Could not Load model from{}'.format(onnx_path)
    #     pass # Error handling code here
    # #-----------------------------------------------------
    # config = builder.create_builder_config()
    # config.max_workspace_size = 256 << 20 # 256 MiB
    # #-----------------------------------------------------
    # serialized_engine = builder.build_serialized_network(network, config)
    # return serialized_engine
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network( 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH )) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
       config.max_workspace_size = (256 << 20) #256 MiB
       #builder.max_batch_size = 1
       if builder.platform_has_fast_fp16 and precision == 'fp16':
           print('*'*10, "___FP16___", '*'*10)
           config.set_flag(trt.BuilderFlag.FP16)

       
       with open(onnx_path, 'rb') as model:
           parser.parse(model.read())
       network.get_input(0).shape = shape
       engine = builder.build_engine(network, config)
       return engine

def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)

       
def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine



if __name__=='__main__':
    engine = build_engine('resnet50.onnx')
    pass