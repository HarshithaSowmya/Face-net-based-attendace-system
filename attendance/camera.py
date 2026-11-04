"""Validate bounded browser-camera uploads before image decoding/inference."""
from io import BytesIO
import warnings

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_FRAME_BYTES = 1024 * 1024
MAX_FRAME_EDGE = 1280


def read_uploaded_frame(request):
    upload = request.FILES.get('frame')
    if upload is None or upload.size > MAX_FRAME_BYTES:
        raise ValueError('Upload one JPEG or PNG frame under 1 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(upload.read())) as image:
                if image.format not in {'JPEG', 'PNG'}:
                    raise ValueError('Only JPEG and PNG images are accepted.')
                width, height = image.size
                if min(width, height) < 32 or max(width, height) > MAX_FRAME_EDGE:
                    raise ValueError('Frame dimensions must be between 32 and 1280 pixels.')
                image.load()
                return cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError('Invalid camera image.') from error
