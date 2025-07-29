"""
Configuration and logging utilities for SDNQ quantization backend.
Replaces functionality from modules.shared.
"""

import logging
import sys
from typing import Optional

# Create logger
logger = logging.getLogger("sdnq")

class SDNQConfig:
    """Configuration options for SDNQ"""
    
    def __init__(self):
        # SDNQ specific options
        # Automatically disable torch.compile in PyInstaller bundles to avoid FileNotFoundError
        # when torch._inductor tries to access files that don't exist in the packaged distribution
        is_frozen = getattr(sys, 'frozen', False)
        self.sdnq_dequantize_compile = not is_frozen  # Enable torch.compile for dequantization (disabled if frozen)
        self.diffusers_offload_mode = "none"  # Offload mode: "none", "sequential", "model"
        
        if is_frozen:
            logger.info("PyInstaller bundle detected - torch.compile disabled for SDNQ to prevent compilation errors")
    
    def set_dequantize_compile(self, enabled: bool):
        """Enable/disable torch.compile for dequantization"""
        self.sdnq_dequantize_compile = enabled
    
    def set_offload_mode(self, mode: str):
        """Set diffusers offload mode"""
        if mode in ["none", "sequential", "model"]:
            self.diffusers_offload_mode = mode
        else:
            raise ValueError(f"Invalid offload mode: {mode}. Must be one of: none, sequential, model")

# Global config instance
opts = SDNQConfig()

class SharedLog:
    """Logging utilities"""
    
    @staticmethod
    def warning(message: str):
        """Log a warning message"""
        logger.warning(message)
    
    @staticmethod 
    def info(message: str):
        """Log an info message"""
        logger.info(message)
    
    @staticmethod
    def error(message: str):
        """Log an error message"""
        logger.error(message)
    
    @staticmethod
    def debug(message: str):
        """Log a debug message"""
        logger.debug(message)

# Global log instance
log = SharedLog()

def configure_logging(level: int = logging.INFO):
    """Configure logging for SDNQ"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ) 