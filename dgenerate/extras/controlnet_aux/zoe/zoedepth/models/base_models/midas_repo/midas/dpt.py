import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .base_model import BaseModel
from . import backbones
from .blocks import FeatureFusionBlock_custom, Interpolate, _make_encoder


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):
        super(DPT, self).__init__()

        self.channels_last = channels_last

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true if you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=None,
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        # Choose trasnformers forward function
        self.forward_transformer = backbones.forward_vit if 'vit' in backbone else backbones.forward_swin
    
    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        try:
            # Original forward pass
            if self.channels_last:
                x.contiguous(memory_format=torch.channels_last)

            # Run the pre-trained backbone
            layers = self.forward_transformer(self.pretrained, x)

            # Reassemble layers into decoder features
            layer_1, layer_2, layer_3, layer_4 = layers

            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)

            path_4 = self.scratch.refinenet4(layer_4_rn)
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

            out = self.scratch.output_conv(path_1)
            
            return out
        except (AttributeError, RuntimeError) as e:
            # Handle compatibility issues
            if "drop_path" in str(e):
                # Try getting the layers with a more resilient method
                # print("Warning: Handling timm compatibility issue in DPT forward pass.")
                # Just reuse the same method but let the transformer function handle any errors
                if self.channels_last:
                    x.contiguous(memory_format=torch.channels_last)

                # The problem is likely in self.forward_transformer, but we've updated it to be compatible
                layers = self.forward_transformer(self.pretrained, x)

                # Continue with the normal process from here
                layer_1, layer_2, layer_3, layer_4 = layers

                layer_1_rn = self.scratch.layer1_rn(layer_1)
                layer_2_rn = self.scratch.layer2_rn(layer_2)
                layer_3_rn = self.scratch.layer3_rn(layer_3)
                layer_4_rn = self.scratch.layer4_rn(layer_4)

                path_4 = self.scratch.refinenet4(layer_4_rn)
                path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
                path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
                path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

                out = self.scratch.output_conv(path_1)
                
                return out
            else:
                # If it's not a known compatibility issue, re-raise
                raise


class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
    ):
        super().__init__(**kwargs)

        self.scale = scale
        self.shift = shift
        self.invert = invert
        self.non_negative = non_negative

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        try:
            # Standard forward pass
            return super().forward(x).squeeze(dim=1)
        except (AttributeError, RuntimeError) as e:
            # Handle compatibility issues
            if "drop_path" in str(e):
                print("Warning: Handling timm compatibility issue. Using alternative forward path.")
                # Manual forward implementation as fallback
                layers = self.forward_transformer(self.pretrained, x)
                layer_1, layer_2, layer_3, layer_4 = layers
                
                layer_1_rn = self.scratch.layer1_rn(layer_1)
                layer_2_rn = self.scratch.layer2_rn(layer_2)
                layer_3_rn = self.scratch.layer3_rn(layer_3)
                layer_4_rn = self.scratch.layer4_rn(layer_4)
                
                path_4 = self.scratch.refinenet4(layer_4_rn)
                path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
                path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
                path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
                
                out = self.scratch.output_conv(path_1)
                
                return out.squeeze(dim=1)
            else:
                # If it's not a known compatibility issue, re-raise
                raise 