# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

import cv2
import numpy as np
import PIL.Image
import spandrel
import torch
from torchvision.transforms.functional import normalize

import dgenerate.hfhub as _hfhub
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.messages as _messages
import dgenerate.types as _types
import dgenerate.webcache as _webcache


def _install_facexlib_downloader(download) -> None:
    """Route facexlib weight downloads through dgenerate's web cache."""
    import facexlib.detection
    import facexlib.parsing

    def load_file_from_url(url, model_dir=None, progress=True, file_name=None, save_dir=None, **kwargs):
        del model_dir, progress, file_name, save_dir, kwargs
        return download(url)

    facexlib.detection.load_file_from_url = load_file_from_url
    facexlib.parsing.load_file_from_url = load_file_from_url


def _create_face_helper(device: torch.device,
                        *,
                        face_size: int,
                        det_model: str):
    from facexlib.detection import retinaface
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper

    if hasattr(retinaface, 'device'):
        retinaface.device = device

    return FaceRestoreHelper(
        upscale_factor=1,
        face_size=face_size,
        crop_ratio=(1, 1),
        det_model=det_model,
        save_ext='png',
        use_parse=True,
        device=device,
    )


def _bgr_uint8_to_rgb_tensor(image_bgr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).to(device=device, dtype=dtype) / 255.0
    return tensor.unsqueeze(0)


def _rgb_tensor_to_bgr_uint8(tensor: torch.Tensor, *, min_max: tuple[float, float]) -> np.ndarray:
    lo, hi = min_max
    tensor = tensor.squeeze(0).float().detach().cpu().clamp_(lo, hi)
    tensor = (tensor - lo) / (hi - lo)
    rgb = tensor.numpy().transpose(1, 2, 0)
    if rgb.shape[2] == 1:
        return np.squeeze(rgb, axis=2).astype('uint8')
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return (bgr * 255.0).astype('uint8')


def _restore_face_crop(descriptor: spandrel.ImageModelDescriptor,
                       crop_bgr: np.ndarray,
                       device: torch.device,
                       dtype: torch.dtype,
                       fidelity: float) -> np.ndarray:
    tensor = _bgr_uint8_to_rgb_tensor(crop_bgr, device, dtype)
    arch_id = descriptor.architecture.id

    if arch_id in {'GFPGAN', 'CodeFormer'}:
        normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        if arch_id == 'GFPGAN':
            restored = descriptor.model(tensor, return_rgb=False)[0]
        else:
            restored = descriptor.model(tensor, weight=fidelity)[0]
        return _rgb_tensor_to_bgr_uint8(restored, min_max=(-1.0, 1.0))

    restored = descriptor(tensor)
    return _rgb_tensor_to_bgr_uint8(restored, min_max=(0.0, 1.0))


def _restore_image_faces(np_image_rgb: np.ndarray,
                         face_helper,
                         descriptor: spandrel.ImageModelDescriptor,
                         restore_device: torch.device,
                         dtype: torch.dtype,
                         fidelity: float,
                         *,
                         only_center_face: bool,
                         eye_dist_threshold: float) -> np.ndarray:
    np_bgr = np_image_rgb[:, :, ::-1].copy()
    original_resolution = np_bgr.shape[0:2]

    face_helper.clean_all()
    try:
        face_helper.read_image(np_bgr)
        face_helper.get_face_landmarks_5(
            only_center_face=only_center_face,
            resize=640,
            eye_dist_threshold=eye_dist_threshold)
        if not face_helper.all_landmarks_5:
            _messages.warning('face-restore: no faces detected; leaving the image unchanged.')
            return np_image_rgb

        face_helper.align_warp_face()
        _messages.debug_log(f'face-restore: restoring {len(face_helper.cropped_faces)} face(s).')

        for cropped_face in face_helper.cropped_faces:
            restored = _restore_face_crop(
                descriptor, cropped_face, restore_device, dtype, fidelity)
            face_helper.add_restored_face(restored)

        face_helper.get_inverse_affine(None)
        merged_bgr = face_helper.paste_faces_to_input_image()
        if original_resolution != merged_bgr.shape[0:2]:
            merged_bgr = cv2.resize(
                merged_bgr,
                (original_resolution[1], original_resolution[0]),
                interpolation=cv2.INTER_LINEAR)
        return merged_bgr[:, :, ::-1]
    finally:
        face_helper.clean_all()


def _place_face_helper_models(face_helper, detector_device: torch.device) -> None:
    face_helper.face_det.to(detector_device)
    face_helper.face_parse.to(detector_device)
    # Parsing during paste follows this device, not the module's current device.
    face_helper.device = detector_device


class FaceRestoreProcessor(_imageprocessor.ImageProcessor):
    """
    Restore detected faces using Spandrel-supported face restoration models.

    This processor detects faces with facexlib (RetinaFace), aligns each face,
    runs a restoration model on the crop, and pastes the results back into the
    image.

    Supported model families include GFPGAN (v1.2 through v1.4), CodeFormer (via
    spandrel_extra_arches), and RestoreFormer. The "model" argument is a local path
    or HTTP/HTTPS URL to a compatible .pth or .safetensors file.

    For example: "face-restore;model=https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

    Chain after Real-ESRGAN in "--post-processors", for example:
    "upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth,face-restore;model=https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

    The "fidelity" argument controls CodeFormer from 0 (quality) to 1 (fidelity to
    the original face). It is ignored for GFPGAN and RestoreFormer.

    Face restoration models often do not support half precision; when "dtype" is
    "float16" and the architecture rejects it, "float32" is used instead.

    The "detector-device" argument may run RetinaFace on a different device than
    the restoration model ("device").

    The "detector" argument selects the facexlib RetinaFace variant, either
    "retinaface_resnet50" (default) or "retinaface_mobile0.25".

    The "face-size" argument sets the aligned crop size in pixels (default 512).
    GFPGAN and CodeFormer require 512. Other sizes must be at least 128 and
    divisible by 32.

    The "eye-dist-threshold" argument filters out small or distant faces (default 5).

    The "only-center-face" argument, when true, restores only the face nearest
    the center of the image.

    The "pre-resize" argument controls whether restoration runs before or after
    dgenerate resizes the image (default: after).

    Example: "face-restore;model=GFPGANv1.4.pth;fidelity=0.7;detector=retinaface_resnet50;dtype=float32"
    """

    NAMES = ['face-restore']

    OPTION_ARGS = {
        'dtype': ['float32', 'float16', 'bfloat16'],
        'detector': ['retinaface_resnet50', 'retinaface_mobile0.25'],
    }

    FILE_ARGS = {
        'model': {
            'mode': 'in',
            'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.ckpt', '*.bin'])],
        },
    }

    def __init__(self,
                 model: str,
                 fidelity: float = 0.5,
                 face_size: int = 512,
                 detector: str = 'retinaface_resnet50',
                 detector_device: _types.OptionalName = None,
                 eye_dist_threshold: float = 5.0,
                 only_center_face: bool = False,
                 dtype: str = 'float32',
                 pre_resize: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        try:
            import facexlib  # noqa: F401
        except ImportError as e:
            raise self.argument_error(
                'The face-restore processor requires the facexlib package.') from e

        if not 0.0 <= fidelity <= 1.0:
            raise self.argument_error('Argument "fidelity" must be between 0 and 1.')

        if face_size < 128 or face_size % 32 != 0:
            raise self.argument_error(
                'Argument "face-size" must be at least 128 and divisible by 32.')

        dtype = dtype.lower()
        if dtype not in self.OPTION_ARGS['dtype']:
            raise self.argument_error(
                f'Argument "dtype" must be one of: {", ".join(self.OPTION_ARGS["dtype"])}.')

        detector = detector.lower()
        if detector not in self.OPTION_ARGS['detector']:
            raise self.argument_error(
                f'Argument "detector" must be one of: {", ".join(self.OPTION_ARGS["detector"])}.')

        self._model_path = model
        self._fidelity = fidelity
        self._face_size = face_size
        self._detector = detector
        self._detector_device_name = detector_device
        self._eye_dist_threshold = eye_dist_threshold
        self._only_center_face = only_center_face
        self._pre_resize = pre_resize
        self._dtype_name = dtype
        self._dtype = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }[dtype]

        def download_facexlib(url):
            try:
                return _hfhub.webcache_or_hf_blob_download(
                    url,
                    local_files_only=self.local_files_only,
                    auth_token=self.auth_token)
            except Exception as e:
                raise self.argument_error(
                    f'Could not download face detection model "{url}": {e}') from e

        _install_facexlib_downloader(download_facexlib)

        self._descriptor = self._load_restoration_model(model, self._dtype)
        # A cached load may have fallen back from half precision. Match the weights.
        self._dtype = self._descriptor.dtype
        self.register_module(self._descriptor)

        arch_id = self._descriptor.architecture.id
        if arch_id in {'GFPGAN', 'CodeFormer'} and self._face_size != 512:
            raise self.argument_error(
                f'Argument "face-size" must be 512 for {arch_id}.')

        if (arch_id != 'CodeFormer' and fidelity != 0.5):
            _messages.debug_log(
                'face-restore: "fidelity" applies to CodeFormer only; ignoring for '
                f'{self._descriptor.architecture.id}.')

        detector_device = torch.device(
            _types.default(self._detector_device_name, self.device))

        def load_helper():
            return _create_face_helper(
                detector_device,
                face_size=self._face_size,
                det_model=self._detector)

        self._face_helper = self.load_object_cached(
            tag=f'facexlib:{detector}:{face_size}:{detector_device}',
            estimated_size=200 * 1024 * 1024,
            method=load_helper)
        self.register_module(self._face_helper.face_det)
        self.register_module(self._face_helper.face_parse)

    def _load_restoration_model(self, model_path: str, dtype: torch.dtype) -> spandrel.ImageModelDescriptor:
        og_path = model_path

        if _webcache.is_downloadable_url(model_path):
            try:
                model_path = _hfhub.webcache_or_hf_blob_download(
                    model_path,
                    local_files_only=self.local_files_only,
                    auth_token=self.auth_token)
            except Exception as e:
                raise self.argument_error(
                    f'Could not download argument "model": "{og_path}", error: {e}') from e

        self.set_size_estimate(os.path.getsize(model_path))

        def load_method():
            try:
                descriptor = spandrel.ModelLoader().load_from_file(model_path).eval()
            except (ValueError, RuntimeError, TypeError, AttributeError) as e:
                raise self.argument_error(f'Unsupported model file format: {e}') from e

            if not isinstance(descriptor, spandrel.ImageModelDescriptor):
                raise self.argument_error(
                    'Unsupported model: face-restore requires a single-image Spandrel model.')

            if descriptor.purpose != 'FaceSR':
                raise self.argument_error(
                    f'Model architecture "{descriptor.architecture.id}" is not a face '
                    f'restoration model (expected purpose FaceSR, got {descriptor.purpose}).')

            dtype_to_use = dtype
            if dtype == torch.float16 and not descriptor.supports_half:
                _messages.warning(
                    f'face-restore: {descriptor.architecture.id} does not support float16; '
                    f'using float32 for "{og_path}".')
                dtype_to_use = torch.float32
                self._dtype = torch.float32
                self._dtype_name = 'float32'
            elif dtype == torch.bfloat16 and not descriptor.supports_bfloat16:
                _messages.warning(
                    f'face-restore: {descriptor.architecture.id} does not support bfloat16; '
                    f'using float32 for "{og_path}".')
                dtype_to_use = torch.float32
                self._dtype = torch.float32
                self._dtype_name = 'float32'

            try:
                descriptor.to(dtype_to_use)
            except spandrel.UnsupportedDtypeError as e:
                raise self.argument_error(
                    f'Argument "dtype" value "{dtype}" is not supported for '
                    f'"{descriptor.architecture.id}" with model "{og_path}".') from e

            return descriptor

        descriptor = self.load_object_cached(
            tag=model_path + self._dtype_name,
            estimated_size=self.size_estimate,
            method=load_method)

        _messages.debug_log(
            f'{_types.fullname(FaceRestoreProcessor._load_restoration_model)}("{model_path}") '
            f'-> {descriptor.architecture.id}')

        return descriptor

    @torch.inference_mode()
    def _process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        self.to(self.device)
        restore_device = self.modules_device
        detector_device = torch.device(
            _types.default(self._detector_device_name, self.device))

        self._descriptor.to(restore_device)
        _place_face_helper_models(self._face_helper, detector_device)

        rgb = np.array(image.convert('RGB'))
        restored = _restore_image_faces(
            rgb,
            self._face_helper,
            self._descriptor,
            restore_device,
            self._dtype,
            self._fidelity,
            only_center_face=self._only_center_face,
            eye_dist_threshold=self._eye_dist_threshold)

        return PIL.Image.fromarray(restored)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize) -> PIL.Image.Image:
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        if not self._pre_resize:
            return self._process(image)
        return image


__all__ = _types.module_all()
