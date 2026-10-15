import os
import time
import pynvml

# 根目录可用环境变量覆盖；默认值是共享旧机 test-ai 上的位置
SATCAST = os.environ.get("SATCAST_DIR", "/data1/satcast")
DCM = os.environ.get("DCM_ROOT", "/home/group1/26fall_aiclass/dcm")
from torch.utils.tensorboard import SummaryWriter

# Initialize NVML and TensorBoard writer
pynvml.nvmlInit()
writer = SummaryWriter(log_dir=f'{DCM}/cira-diff/runs/models/latentedm_10_two_inputs_radames_fixeddata_TEST/logs/gpu_stats/')

def log_gpu_stats():
    device_count = pynvml.nvmlDeviceGetCount()
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        writer.add_scalar(f'GPU_{i}/Usage', gpu_utilization, time.time())
        writer.add_scalar(f'GPU_{i}/Temperature', temperature, time.time())

if __name__ == "__main__":
    while True:
        log_gpu_stats()
        time.sleep(30)

    # Clean up NVML
    pynvml.nvmlShutdown()