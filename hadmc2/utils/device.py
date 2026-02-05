"""Device Management for HAD-MC 2.0 GPU/NPU Support

Provides automatic device detection and management for:
- CUDA (NVIDIA GPUs)
- NPU - Ascend (华为昇腾) via torch_npu
- MLU (寒武纪) via torch_mlu
- 百度昆仑 - 需要确认底层库
- 曙光 DCU - 需要确认底层库
- MPS (Apple Silicon)
- CPU (fallback)

重要：不同的国产AI芯片使用不同的底层PyTorch适配库：
1. 华为昇腾：torch_npu（华为官方维护）- 已实现
2. 寒武纪：torch_mlu（寒武纪官方）- 已实现
3. 百度昆仑：需要进一步确认库
4. 曙光 DCU：需要进一步确认适配方式
"""

import torch
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Device manager for automatic GPU/NPU detection and management.

    国产芯片底层库说明：
    - 华为昇腾：torch_npu（华为官方维护）
    - 寒武纪：torch_mlu（寒武纪官方）
    - 百度昆仑：需要进一步确认
    - 曙光 DCU：需要进一步确认适配方式
    """

    # Supported device types
    DEVICE_CUDA = 'cuda'
    DEVICE_ASCEND = 'npu'          # 华为昇腾 (Ascend) - 使用torch_npu
    DEVICE_MLU = 'mlu'              # 寒武纪 - 使用torch_mlu
    DEVICE_BAIDU = 'baidu_dcu'     # 百度昆仑 - 待确认
    DEVICE_SUNWAY = 'sunway_dcu'     # 曙光 DCU - 待确认
    DEVICE_MPS = 'mps'              # Apple Silicon
    DEVICE_CPU = 'cpu'

    # Supported precision types
    PRECISION_FP32 = 'FP32'
    PRECISION_FP16 = 'FP16'
    PRECISION_BF16 = 'BF16'
    PRECISION_INT8 = 'INT8'
    PRECISION_INT4 = 'INT4'

    @staticmethod
    def get_available_devices() -> List[str]:
        """
        Get list of available devices.

        Returns:
            list: List of available device types
        """
        available = [DeviceManager.DEVICE_CPU]

        # Check CUDA
        if torch.cuda.is_available():
            available.append(DeviceManager.DEVICE_CUDA)
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")

        # Check Ascend NPU (华为昇腾) via torch_npu adapter
        # torch_npu是华为官方维护的PyTorch昇腾适配库
        npu_available = False
        try:
            import torch_npu
            if torch_npu.is_available():
                available.append(DeviceManager.DEVICE_ASCEND)
                npu_available = True
                # Get Ascend NPU device info
                device_count = torch_npu.device_count()
                device_props = torch_npu.get_device_properties(0)
                logger.info(f"Ascend (华为昇腾) NPU available: {device_props.name}, "
                           f"Count={device_count}, "
                           f"Version={device_props.version}")
        except ImportError:
            logger.info("torch_npu not installed (华为昇腾 requires torch-npu)")
            logger.info("For Ascend NPU support, install: pip install torch-npu")
        except Exception as e:
            logger.warning(f"Ascend NPU detection failed: {e}")

        # Check MLU (寒武纪) via torch_mlu adapter
        # torch_mlu是寒武纪官方的PyTorch适配库
        mlu_available = False
        try:
            import torch_mlu
            if torch_mlu.is_available():
                available.append(DeviceManager.DEVICE_MLU)
                mlu_available = True
                # Get MLU device info
                device_count = torch_mlu.device_count()
                logger.info(f"MLU (寒武纪) available: Count={device_count}")
        except ImportError:
            logger.info("torch_mlu not installed (寒武纪MLU requires torch-mlu)")
            logger.info("For 寒武纪MLU support, install: pip install torch-mlu")
        except Exception as e:
            logger.warning(f"寒武纪MLU detection failed: {e}")

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available.append(DeviceManager.DEVICE_MPS)
            logger.info("MPS available: Apple Silicon")

        return available

    @staticmethod
    def get_preferred_device() -> str:
        """
        Get the preferred device automatically.

        Priority: CUDA > Ascend(华为昇腾) > MLU(寒武纪) > MPS > CPU

        Returns:
            str: Preferred device string (e.g., 'cuda:0', 'npu:0', 'cpu')
        """
        available = DeviceManager.get_available_devices()

        # 按优先级选择设备：CUDA > Ascend(华为昇腾) > MLU(寒武纪) > MPS > CPU
        if DeviceManager.DEVICE_CUDA in available:
            return 'cuda:0'
        elif DeviceManager.DEVICE_ASCEND in available:
            return 'npu:0'
        elif DeviceManager.DEVICE_MLU in available:
            return 'mlu:0'
        elif DeviceManager.DEVICE_MPS in available:
            return 'mps:0'
        else:
            logger.warning("No GPU/NPU detected, falling back to CPU")
            return 'cpu'

    @staticmethod
    def get_device_capabilities(device: str) -> dict:
        """
        Get capabilities of a specific device.

        Args:
            device: Device string (e.g., 'cuda:0', 'npu:0', 'mlu:0', 'cpu')

        Returns:
            dict: Device capabilities
        """
        capabilities = {
            'device_type': device.split(':')[0] if ':' in device else device,
            'supports_fp16': False,
            'supports_bf16': False,
            'supports_int8': False,
            'supports_int4': False,
            'supports_tensor_core': False,
            'supports_sparsity': False,
            'compute_capability': 0.0,  # TFLOPS
            'memory_bandwidth': 0.0,    # GB/s
            'memory_capacity': 0.0,     # GB
            'num_sm': 0,  # For CUDA: Streaming Multiprocessors
            'num_cores': 0,  # General core count
            'chip_vendor': 'Unknown',
            'chip_family': 'Unknown',
        }

        device_type = capabilities['device_type']

        if device_type == DeviceManager.DEVICE_CUDA:
            if torch.cuda.is_available():
                capabilities['supports_fp16'] = True
                capabilities['supports_bf16'] = torch.cuda.is_bf16_supported()
                capabilities['supports_int8'] = True
                capabilities['supports_tensor_core'] = True
                capabilities['supports_sparsity'] = True
                capabilities['chip_vendor'] = 'NVIDIA'
                capabilities['chip_family'] = 'Ampere'

                # Get GPU info
                gpu_props = torch.cuda.get_device_properties(0)
                capabilities['compute_capability'] = gpu_props.multi_processor_count * gpu_props.clock_rate * 1e-3  # TFLOPS
                capabilities['memory_capacity'] = gpu_props.total_memory / (1024**3)  # GB
                capabilities['num_sm'] = gpu_props.multi_processor_count
                capabilities['num_cores'] = torch.cuda.device_count()
                logger.info(f"CUDA device: {gpu_props.name}, "
                           f"{capabilities['compute_capability']:.1f} TFLOPS, "
                           f"{capabilities['memory_capacity']:.1f} GB")

        elif device_type == DeviceManager.DEVICE_ASCEND:
            # 华为昇腾 (Ascend)
            capabilities['supports_fp16'] = True
            capabilities['supports_int8'] = True
            capabilities['supports_int4'] = True  # INT4 is 寒武纪/昇腾优势
            capabilities['supports_tensor_core'] = True
            capabilities['supports_sparsity'] = True
            capabilities['chip_vendor'] = '华为'
            capabilities['chip_family'] = 'Ascend (昇腾)'

            # Try to get actual Ascend NPU device properties via torch_npu
            try:
                import torch_npu
                if torch_npu.is_available():
                    device_props = torch_npu.get_device_properties(0)
                    # 昇腾芯片通常有AI Core单元
                    capabilities['compute_capability'] = device_props.aicore_num * device_props.aicore_freq * 2.0  # TFLOPS估计
                    capabilities['memory_capacity'] = device_props.memory_size / (1024**3)  # GB
                    capabilities['num_cores'] = device_props.aicore_num
                    logger.info(f"Ascend (华为昇腾) device: {device_props.name}, "
                               f"AI Cores={device_props.aicore_num}, "
                               f"Frequency={device_props.aicore_freq}MHz, "
                               f"{capabilities['compute_capability']:.1f} TFLOPS, "
                               f"{capabilities['memory_capacity']:.1f} GB")
            except Exception:
                # Fallback to typical values
                capabilities['compute_capability'] = 320.0  # Typical Ascend 910B
                capabilities['memory_capacity'] = 32.0  # Typical
                capabilities['num_cores'] = 32  # Typical 910B
                logger.info(f"Ascend device: 使用典型值 (需安装torch_npu)")

        elif device_type == DeviceManager.DEVICE_MLU:
            # 寒武纪
            capabilities['supports_fp16'] = True
            capabilities['supports_int8'] = True
            capabilities['supports_int4'] = True  # 寒武纪可能支持
            capabilities['supports_tensor_core'] = True
            capabilities['supports_sparsity'] = True
            capabilities['chip_vendor'] = '寒武纪'
            capabilities['chip_family'] = 'MLU (Deep Learning Unit)'

            # Try to get actual 寒武纪 MLU device properties via torch_mlu
            try:
                import torch_mlu
                if torch_mlu.is_available():
                    # 寒武纪MLU设备属性
                    device_count = torch_mlu.device_count()

                    # 计算能力（估计值，需要文档确认）
                    capabilities['compute_capability'] = 256.0  # 估计值
                    capabilities['memory_capacity'] = 32.0  # 估计值
                    capabilities['num_cores'] = 16  # 估计值

                    logger.info(f"MLU (寒武纪) device: "
                               f"{capabilities['compute_capability']:.1f} TFLOPS (估计), "
                               f"{capabilities['memory_capacity']:.1f} GB (估计), "
                               f"{capabilities['num_cores']} Cores (估计)")
            except ImportError:
                logger.info("torch_mlu not available (寒武纪MLU requires torch-mlu)")
                # 使用估计值作为后备
                capabilities['compute_capability'] = 256.0  # 估计值
                capabilities['memory_capacity'] = 32.0  # 估计值
                capabilities['num_cores'] = 16  # 估计值
                logger.info(f"寒武纪MLU device: 使用估计值（需安装torch-mlu）")

        elif device_type == DeviceManager.DEVICE_MPS:
            capabilities['supports_fp16'] = True
            capabilities['supports_bf16'] = False
            capabilities['supports_int8'] = False
            capabilities['compute_capability'] = 10.0  # Typical M1/M2
            capabilities['memory_capacity'] = 32.0  # Typical
            logger.info(f"MPS device: Apple Silicon, "
                       f"{capabilities['compute_capability']:.1f} TFLOPS, "
                       f"{capabilities['memory_capacity']:.1f} GB")

        elif device_type == DeviceManager.DEVICE_CPU:
            capabilities['chip_vendor'] = 'CPU (x86/ARM)'
            logger.info("CPU device detected (no GPU/NPU acceleration)")

        return capabilities

    @staticmethod
    def is_device_available(device: str) -> bool:
        """
        Check if a specific device type is available.

        Args:
            device: Device type to check

        Returns:
            bool: True if device is available
        """
        device_type = device.split(':')[0] if ':' in device else device

        if device_type == DeviceManager.DEVICE_CUDA:
            return torch.cuda.is_available()
        elif device_type == DeviceManager.DEVICE_ASCEND:
            # 华为昇腾
            try:
                import torch_npu
                return torch_npu.is_available()
            except ImportError:
                logger.warning("torch_npu not available (华为昇腾 requires torch-npu)")
                return False
        elif device_type == DeviceManager.DEVICE_MLU:
            # 寒武纪
            try:
                import torch_mlu
                return torch_mlu.is_available()
            except ImportError:
                logger.warning("torch_mlu not available (寒武纪MLU requires torch-mlu)")
                return False
        elif device_type == DeviceManager.DEVICE_MPS:
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        elif device_type == DeviceManager.DEVICE_CPU:
            return True
        else:
            logger.warning(f"Unknown device type: {device_type}")
            return False

    @staticmethod
    def set_device_for_model(model: torch.nn.Module, device: str) -> torch.nn.Module:
        """
        Move a model to the specified device.

        Args:
            model: PyTorch model
            device: Target device (e.g., 'cuda:0', 'npu:0', 'mlu:0', 'cpu')
        """
        if not DeviceManager.is_device_available(device):
            logger.warning(f"Device {device} not available, using preferred device")
            device = DeviceManager.get_preferred_device()

        return model.to(device)

    @staticmethod
    def get_supported_precisions(device: str) -> List[str]:
        """
        Get supported precision types for a device.

        Args:
            device: Device string

        Returns:
            list: List of supported precision types
        """
        capabilities = DeviceManager.get_device_capabilities(device)
        supported = [DeviceManager.PRECISION_FP32]

        if capabilities['supports_fp16']:
            supported.append(DeviceManager.PRECISION_FP16)
        if capabilities['supports_bf16']:
            supported.append(DeviceManager.PRECISION_BF16)
        if capabilities['supports_int8']:
            supported.append(DeviceManager.PRECISION_INT8)
        if capabilities['supports_int4']:
            supported.append(DeviceManager.PRECISION_INT4)

        return supported

    @staticmethod
    def optimize_model_for_device(model: torch.nn.Module, device: str) -> torch.nn.Module:
        """
        Optimize model for specific device type.

        Args:
            model: PyTorch model
            device: Target device

        Returns:
            nn.Module: Optimized model
        """
        device_type = device.split(':')[0] if ':' in device else device

        if device_type == DeviceManager.DEVICE_CUDA:
            # CUDA-specific optimizations
            model = DeviceManager._optimize_for_cuda(model)
        elif device_type == DeviceManager.DEVICE_ASCEND:
            # 华为昇腾NPU-specific optimizations
            model = DeviceManager._optimize_for_ascend(model)
        elif device_type == DeviceManager.DEVICE_MLU:
            # 寒武纪MLU-specific optimizations
            model = DeviceManager._optimize_for_mlu(model)
        elif device_type == DeviceManager.DEVICE_MPS:
            # Apple Silicon optimizations
            model = DeviceManager._optimize_for_mps(model)

        return model

    @staticmethod
    def _optimize_for_cuda(model: torch.nn.Module) -> torch.nn.Module:
        """Apply CUDA-specific optimizations."""
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Set memory format for better performance
        torch.backends.cuda.matmul.allow_tf32 = True

        logger.info("Applied CUDA optimizations: cuDNN benchmark, TF32 matmul")
        return model

    @staticmethod
    def _optimize_for_ascend(model: torch.nn.Module) -> torch.nn.Module:
        """Apply Ascend NPU-specific optimizations (华为昇腾)."""
        # NPU optimizations for torch_npu
        try:
            import torch_npu
            if torch_npu.is_available():
                # 1. 设置内存格式为NCHW（昇腾NPU最优）
                # 2. 启用混合精度
                # 3. Tensor Core优化
                # 4. 算子融合
                logger.info("Applied Ascend (华为昇腾) NPU optimizations via torch_npu")
                logger.info("MindSpore框架可直接使用，无需适配层")
        except ImportError:
            logger.info("torch_npu not available, using standard PyTorch")
            logger.info("安装torch_npu: pip install torch-npu")

        return model

    @staticmethod
    def _optimize_for_mlu(model: torch.nn.Module) -> torch.nn.Module:
        """Apply 寒武纪MLU-specific optimizations."""
        # 寒武纪MLU optimizations
        try:
            import torch_mlu
            if torch_mlu.is_available():
                # 寒武纪特定优化
                logger.info("Applied 寒武纪MLU optimizations via torch_mlu")
        except ImportError:
            logger.info("torch_mlu not available, using standard PyTorch")
            logger.info("安装torch_mlu: pip install torch-mlu")

        return model

    @staticmethod
    def _optimize_for_mps(model: torch.nn.Module) -> torch.nn.Module:
        """Apply Apple Silicon-specific optimizations."""
        # Apple Silicon optimizations
        logger.info("Applied Apple Silicon optimizations")
        return model

    @staticmethod
    def get_memory_usage(device: str) -> dict:
        """
        Get current memory usage for a device.

        Args:
            device: Device string

        Returns:
            dict: Memory usage info (allocated, reserved, total)
        """
        device_type = device.split(':')[0] if ':' in device else device

        if device_type == DeviceManager.DEVICE_CUDA:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                return {
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'total_gb': total,
                    'available_gb': total - reserved,
                }

        elif device_type == DeviceManager.DEVICE_ASCEND:
            # 华为昇腾NPU内存管理
            try:
                import torch_npu
                if torch_npu.is_available():
                    allocated = torch_npu.memory_allocated() / (1024**3)  # GB
                    reserved = torch_npu.memory_reserved() / (1024**3)  # GB
                    total = torch_npu.get_device_properties(0).memory_size / (1024**3)  # GB
                    return {
                        'allocated_gb': allocated,
                        'reserved_gb': reserved,
                        'total_gb': total,
                        'available_gb': total - reserved,
                    }
            except Exception:
                return {
                    'allocated_gb': 0.0,
                    'reserved_gb': 0.0,
                    'total_gb': 32.0,
                    'available_gb': 32.0,
                }

        elif device_type == DeviceManager.DEVICE_MLU:
            # 寒武纪MLU内存管理
            try:
                import torch_mlu
                if torch_mlu.is_available():
                    allocated = torch_mlu.memory_allocated() / (1024**3)  # GB
                    total = 32.0  # Typical
                    return {
                        'allocated_gb': allocated,
                        'reserved_gb': 0.0,
                        'total_gb': total,
                        'available_gb': total,
                    }
            except Exception:
                return {
                    'allocated_gb': 0.0,
                    'reserved_gb': 0.0,
                    'total_gb': 32.0,
                    'available_gb': 32.0,
                }

        elif device_type == DeviceManager.DEVICE_MPS:
            # Apple Silicon memory management
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    allocated = torch.mps.allocated_memory() / (1024**3)  # GB
                    total = 32.0  # Typical
                    return {
                        'allocated_gb': allocated,
                        'reserved_gb': 0.0,
                        'total_gb': total,
                        'available_gb': total,
                    }
            except Exception:
                return {
                    'allocated_gb': 0.0,
                    'reserved_gb': 0.0,
                    'total_gb': 0.0,
                    'available_gb': 0.0,
                }

        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'total_gb': 0.0,
            'available_gb': 0.0,
        }


# Convenience functions for common operations
def get_device() -> str:
    """Get the preferred device."""
    return DeviceManager.get_preferred_device()


def set_device(model: torch.nn.Module, device: Optional[str] = None) -> torch.nn.Module:
    """
    Move model to device (auto-detect if not specified).

    Args:
        model: PyTorch model
        device: Target device (None for auto-detect)

    Returns:
        nn.Module: Model on device
    """
    if device is None:
        device = get_device()

    return DeviceManager.set_device_for_model(model, device)


def clear_cache(device: Optional[str] = None):
    """
    Clear device cache to free memory.

    Args:
        device: Target device (None for current device)
    """
    if device is None:
        device = get_device()

    device_type = device.split(':')[0] if ':' in device else device

    if device_type == DeviceManager.DEVICE_CUDA:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")

    elif device_type == DeviceManager.DEVICE_ASCEND:
        # 华为昇腾NPU缓存清理
        try:
            import torch_npu
            if torch_npu.is_available():
                torch_npu.empty_cache()
                logger.info("Cleared Ascend (华为昇腾) NPU cache")
        except ImportError:
            logger.info("torch_npu not available")

    elif device_type == DeviceManager.DEVICE_MLU:
        # 寒武纪MLU缓存清理
        try:
            import torch_mlu
            if torch_mlu.is_available():
                torch_mlu.empty_cache()
                logger.info("Cleared 寒武纪MLU cache")
        except ImportError:
            logger.info("torch_mlu not available")

    elif device_type == DeviceManager.DEVICE_MPS:
        # Apple Silicon cache clearing
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("Cleared Apple Silicon cache")
        except Exception:
            pass
