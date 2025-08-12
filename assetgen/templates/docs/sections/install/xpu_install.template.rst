Install with XPU support
========================

In order to install with XPU support (Intel), follow the install guide for
your respective platform, but use the XPU-specific PyTorch wheel index:

``--extra-index-url https://download.pytorch.org/whl/xpu``

This will ensure you get XPU-optimized versions of PyTorch with built-in Intel XPU support. 
No additional Intel Extension packages are required - XPU support is included directly 
in the PyTorch XPU wheel.

This is supported experimentally.