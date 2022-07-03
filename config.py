import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import os 
print_log_to_console=True
def get_print_log_to_console():
    return print_log_to_console
def enable_print_log_to_console():
    global print_log_to_console
    print_log_to_console=True
def disable_print_log_to_console():
    global print_log_to_console
    print_log_to_console=False
def enable_gpu_momory_dynamic_growth():
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
def enable_mixed_precision_policy():
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
def disable_tensorflow_warning():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def enable_cuda_visible_devices(devices):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=devices