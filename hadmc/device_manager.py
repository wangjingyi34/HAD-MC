"""
HAD-MC设备管理器 - 自动检测和切换CPU/NPU

支持的NPU:
- 寒武纪MLU (MLU370, MLU590)
- 华为昇腾 (Ascend 910, 910B, 310P)
- CPU (作为fallback)
"""

import os
import logging
from typing import Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceType(Enum):
    CPU = "cpu"
    MLU = "mlu"
    ASCEND = "ascend"
    CUDA = "cuda"

class DeviceManager:
    def __init__(self, prefer_device: Optional[str] = None):
        self.prefer_device = prefer_device
        self.current_device = None
        self.device_info = {}
        self._detect_devices()
        self._select_device()
        
    def _detect_devices(self):
        logger.info("检测可用设备...")
        self._detect_mlu()
        self._detect_ascend()
        self._detect_cpu()
        
    def _detect_mlu(self):
        try:
            import torch_mlu
            import torch
            if torch.mlu.is_available():
                self.device_info[DeviceType.MLU] = {'available': True}
                logger.info("✅ 检测到寒武纪MLU")
                return
        except:
            pass
        self.device_info[DeviceType.MLU] = {'available': False}
        logger.info("❌ 未检测到寒武纪MLU")
        
    def _detect_ascend(self):
        try:
            import torch_npu
            import torch
            if torch.npu.is_available():
                self.device_info[DeviceType.ASCEND] = {'available': True}
                logger.info("✅ 检测到华为昇腾")
                return
        except:
            pass
        self.device_info[DeviceType.ASCEND] = {'available': False}
        logger.info("❌ 未检测到华为昇腾")
        
    def _detect_cpu(self):
        self.device_info[DeviceType.CPU] = {'available': True}
        logger.info("✅ CPU可用")
        
    def _select_device(self):
        priority = [DeviceType.MLU, DeviceType.ASCEND, DeviceType.CPU]
        for device_type in priority:
            if self.device_info.get(device_type, {}).get('available', False):
                self.current_device = device_type
                logger.info(f"✅ 选择设备: {device_type.value}")
                return
                
    def get_device(self) -> str:
        if self.current_device == DeviceType.CPU:
            return 'cpu'
        elif self.current_device == DeviceType.MLU:
            return 'mlu:0'
        elif self.current_device == DeviceType.ASCEND:
            return 'npu:0'
        return 'cpu'
        
    def is_npu(self) -> bool:
        return self.current_device in [DeviceType.MLU, DeviceType.ASCEND]

_global_dm = None

def get_device_manager():
    global _global_dm
    if _global_dm is None:
        _global_dm = DeviceManager()
    return _global_dm

def get_device():
    return get_device_manager().get_device()
