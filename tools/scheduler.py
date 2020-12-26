import subprocess

from loguru import logger


def _get_gpu_info():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.free',
                                      '--format=csv,noheader']).decode("utf-8")
    result = result.strip().split('\n')
    gpus_info = [info.split(',') for info in result]
    extract_digits = lambda x: int(''.join(filter(str.isdigit, x)))
    gpus_info = [list(map(extract_digits, info)) for info in gpus_info]
    gpus_info = [[idx] + info for idx, info in enumerate(gpus_info)]
    gpus_info = [dict(zip(['id', 'utilization', 'memory'], info)) for info in gpus_info]
    return gpus_info


def get_gpu_id(min_memory=1000):
    gpus_info = _get_gpu_info()
    gpus_info = [info for info in gpus_info if info['memory'] > min_memory]
    # The first one will have the most free space
    gpus_info.sort(key=lambda x: -x['memory'])
    # The first one will have the least utilization
    gpus_info.sort(key=lambda x: x['utilization'])
    gpu_id = gpus_info[0]['id']
    logger.info(f"GPU {gpu_id} was scheduled")
    return gpu_id
