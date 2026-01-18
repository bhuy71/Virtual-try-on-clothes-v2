"""
Inference module for Virtual Try-On models.

This module provides easy-to-use inference pipelines for both
CP-VTON (traditional) and HR-VITON (state-of-the-art) methods.
"""

from .inference import VITONInference

__all__ = ['VITONInference']
