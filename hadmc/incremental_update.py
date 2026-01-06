"""Algorithm 5: Hash-based Incremental Update"""
from .device_manager import DeviceManager

import torch
import hashlib
import logging

logger = logging.getLogger(__name__)


class HashBasedUpdater:
    """Implements Algorithm 5 from HAD-MC paper"""
    
    def __init__(self, block_size=4096):
        self.block_size = block_size
        
    def divide_into_blocks(self, model):
        """Divide model parameters into blocks"""
        blocks = []
        state_dict = model.state_dict()
        
        for name, param in state_dict.items():
            # Flatten parameter
            flat_param = param.flatten()
            
            # Divide into blocks
            num_elements = flat_param.numel()
            num_blocks = (num_elements + self.block_size - 1) // self.block_size
            
            for i in range(num_blocks):
                start = i * self.block_size
                end = min((i + 1) * self.block_size, num_elements)
                block_data = flat_param[start:end]
                blocks.append({
                    'layer': name,
                    'block_id': i,
                    'data': block_data
                })
        
        logger.info(f"Divided model into {len(blocks)} blocks")
        return blocks
    
    def compute_hash(self, block_data):
        """Compute SHA256 hash of a block"""
        # Convert tensor to bytes
        bytes_data = block_data.cpu().numpy().tobytes()
        hash_obj = hashlib.sha256(bytes_data)
        return hash_obj.hexdigest()
    
    def compare_models(self, model_old, model_new):
        """Compare two models and find changed blocks"""
        blocks_old = self.divide_into_blocks(model_old)
        blocks_new = self.divide_into_blocks(model_new)
        
        if len(blocks_old) != len(blocks_new):
            logger.warning("Models have different number of blocks")
            return blocks_new, len(blocks_new)
        
        changed_blocks = []
        for b_old, b_new in zip(blocks_old, blocks_new):
            hash_old = self.compute_hash(b_old['data'])
            hash_new = self.compute_hash(b_new['data'])
            
            if hash_old != hash_new:
                changed_blocks.append(b_new)
        
        logger.info(f"Changed blocks: {len(changed_blocks)}/{len(blocks_new)}")
        return changed_blocks, len(blocks_new)
    
    def create_update_package(self, model_old, model_new):
        """Create incremental update package"""
        changed_blocks, total_blocks = self.compare_models(model_old, model_new)
        
        # Calculate bandwidth reduction
        bandwidth_reduction = 1 - (len(changed_blocks) / total_blocks)
        
        logger.info(f"Bandwidth reduction: {bandwidth_reduction*100:.1f}%")
        
        return {
            'changed_blocks': changed_blocks,
            'bandwidth_reduction': bandwidth_reduction
        }
    
    def run(self, model_old, model_new):
        """Run complete Algorithm 5"""
        update_package = self.create_update_package(model_old, model_new)
        return update_package


# Alias for backward compatibility
class IncrementalUpdater(HashBasedUpdater):
    """Alias for HashBasedUpdater for backward compatibility"""
    pass
