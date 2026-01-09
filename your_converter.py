# your_converter.py
# OCR + mask + LaMa inpainting core logic

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
from PIL import Image

from paddleocr import PaddleOCR  # PaddleOCR Python API [web:152]
from simple_lama_inpainting import SimpleLama  # expects binary mask 255=inpaint [web:153][web:73]


@dataclass
class TextRegion:
    # bbox: (x, y, w, h) in image pixel coordinates
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float
    # quad: 4-point polygon from PaddleOCR (optional)
    quad: Optional[List[List[float]]] = None


class EditableDocConverter:
    """
    Convert an image page into:
    - clean background image (text removed by inpainting)
    - list of OCR text regions for overlay (PPTX/HTML)
    """

    def __init__(
        self,
        lang: str = "ch",
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        min_confidence: float = 0.50,
    ):
        # Initialize OCR once (downloads models on first run) [web:152]
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_gpu=use_gpu)  # [web:152]
        self.inpainter = SimpleLama()  # [web:153]
        self.min_confidence = float(min_confidence)

    def process_document(
        self,
        image_path: str,
        clean_image_path: Optional[str] = None,
        dilation_size: int = 5,
        dilation_iter: int = 2,
        return_mask: bool = False,
    ):
        """
        Args:
            image_path: input PNG/JPG
            clean_image_path: if provided, save cleaned background
            dilation_size: mask dilation kernel size (odd recommended)
            dilation_iter: dilation iterations
            return_mask: if True, also return mask image (PIL)

        Returns:
            clean_pil: PIL.Image (RGB)
            text_regions: List[Dict] (serializable for Streamlit)
            (optional) mask_pil: PIL.Image (L)
        """
        image_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(image_pil)

        # 1) OCR detect + recognize [web:152]
        ocr_result = self.ocr.ocr(img_np, cls=True)  # [web:152]
        regions = self._extract_text_regions(ocr_result)

        # 2) Create binary mask (255=inpaint) for LaMa [web:153][web:73]
        mask_np = self._create_binary_mask(
            img_shape=img_np.shape,
            regions=regions,
            dilation_size=dilation_size,
            dilation_iter=dilation_iter,
        )
        mask_pil = Image.fromarray(mask_np).convert("L")

        # 3) Inpaint
        clean_pil = self.inpainter(image_pil, mask_pil)  # [web:153]

        if clean_image_path:
            clean_pil.save(clean_image_path)

        text_regions = [
            {
                "bbox": r.bbox,
                "text": r.text,
                "confidence": r.confidence,
                "quad": r.quad,
            }
            for r in regions
        ]

        if return_mask:
            return clean_pil, text_regions, mask_pil
        return clean_pil, text_regions

    def _extract_text_regions(self, ocr_result) -> List[TextRegion]:
        """
        PaddleOCR result per line: [quad_points, (text, confidence)] [web:152]
        """
        regions: List[TextRegion] = []

        if not ocr_result or not ocr_result[0]:
            return regions

        for line in ocr_result[0]:
            quad = line[0]                 # 4 points polygon [web:152]
            text = line[1][0]
            conf = float(line[1][1])

            if conf < self.min_confidence:
                continue
            if not text or not text.strip():
                continue

            # Convert quad -> axis-aligned bbox
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            x0, y0 = int(max(0, min(xs))), int(max(0, min(ys)))
            x1, y1 = int(max(xs)), int(max(ys))
            w, h = max(1, x1 - x0), max(1, y1 - y0)

            regions.append(
                TextRegion(
                    bbox=(x0, y0, w, h),
                    text=text,
                    confidence=conf,
                    quad=[[float(p[0]), float(p[1])] for p in quad],
                )
            )

        return regions

    def _create_binary_mask(
        self,
        img_shape: Tuple[int, int, int],
        regions: List[TextRegion],
        dilation_size: int = 5,
        dilation_iter: int = 2,
    ) -> np.ndarray:
        """
        Output: uint8 mask (H,W), 0 keep, 255 inpaint [web:153][web:73]
        """
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Paint text regions white (255)
        for r in regions:
            x, y, bw, bh = r.bbox
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            cv2.rectangle(mask, (x, y), (x2, y2), 255, thickness=-1)

        # Dilation to cover anti-aliased edges & shadows
        k = int(max(1, dilation_size))
        if k % 2 == 0:
            k += 1  # make it odd
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        iters = int(max(0, dilation_iter))
        if iters > 0:
            mask = cv2.dilate(mask, kernel, iterations=iters)

        return mask
