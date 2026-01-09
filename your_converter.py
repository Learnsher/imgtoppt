# your_converter.py (no cv2 dependency)

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw

from paddleocr import PaddleOCR
from simple_lama_inpainting import SimpleLama


@dataclass
class TextRegion:
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float
    quad: Optional[List[List[float]]] = None


class EditableDocConverter:
    def __init__(
        self,
        lang: str = "ch",
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        min_confidence: float = 0.50,
    ):
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_gpu=use_gpu)
        self.inpainter = SimpleLama()
        self.min_confidence = float(min_confidence)

    def process_document(
        self,
        image_path: str,
        clean_image_path: Optional[str] = None,
        dilation_size: int = 5,
        dilation_iter: int = 2,
        return_mask: bool = False,
    ):
        image_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(image_pil)

        # OCR
        ocr_result = self.ocr.ocr(img_np, cls=True)
        regions = self._extract_text_regions(ocr_result)

        # Create mask (PIL version, no cv2)
        mask_pil = self._create_binary_mask_pil(
            img_size=image_pil.size,
            regions=regions,
            padding=dilation_size,  # simple padding instead of morphological dilation
        )

        # Inpaint
        clean_pil = self.inpainter(image_pil, mask_pil)

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
        regions: List[TextRegion] = []
        if not ocr_result or not ocr_result[0]:
            return regions

        for line in ocr_result[0]:
            quad = line[0]
            text = line[1][0]
            conf = float(line[1][1])

            if conf < self.min_confidence or not text.strip():
                continue

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

    def _create_binary_mask_pil(
        self,
        img_size: Tuple[int, int],
        regions: List[TextRegion],
        padding: int = 5,
    ) -> Image.Image:
        """
        Create mask using PIL (no cv2 dependency)
        padding: simple expand bbox by N pixels (replaces cv2 dilation)
        """
        w, h = img_size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for r in regions:
            x, y, bw, bh = r.bbox
            # expand bbox by padding
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(w, x + bw + padding)
            y1 = min(h, y + bh + padding)
            draw.rectangle([x0, y0, x1, y1], fill=255)

        return mask


# ---- PPTX Exporter ----
from pptx import Presentation
from pptx.util import Inches, Pt


class PPTXExporter:
    def __init__(self, slide_width_in=13.333, slide_height_in=7.5):
        self.prs = Presentation()
        self.prs.slide_width = Inches(slide_width_in)
        self.prs.slide_height = Inches(slide_height_in)

    def add_slide_with_overlay(self, bg_image_path, text_regions):
        blank_layout = self.prs.slide_layouts[6]
        slide = self.prs.slides.add_slide(blank_layout)

        img = Image.open(bg_image_path)
        img_w, img_h = img.size
        slide_w = self.prs.slide_width
        slide_h = self.prs.slide_height

        scale = max(slide_w / img_w, slide_h / img_h)
        pic_w = int(img_w * scale)
        pic_h = int(img_h * scale)
        left = int((slide_w - pic_w) / 2)
        top = int((slide_h - pic_h) / 2)

        slide.shapes.add_picture(bg_image_path, left, top, pic_w, pic_h)

        scale_x = slide_w / img_w
        scale_y = slide_h / img_h

        for r in text_regions:
            x, y, w, h = r["bbox"]
            text = r["text"]

            tb = slide.shapes.add_textbox(
                int(x * scale_x),
                int(y * scale_y),
                max(1, int(w * scale_x)),
                max(1, int(h * scale_y)),
            )
            tf = tb.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = text

            font_size = max(8, int(h * scale_y * 0.7))
            p.font.size = Pt(font_size)

            tb.line.fill.background()
            tb.fill.backgrou
