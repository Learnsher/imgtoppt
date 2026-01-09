import easyocr

class EditableDocConverter:
    def __init__(self, lang='ch', ...):
        # 改用 EasyOCR
        self.ocr = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        self.inpainter = SimpleLama()
        
    def _extract_text_regions(self, img_np):
        # EasyOCR 返回格式：[[bbox, text, confidence], ...]
        results = self.ocr.readtext(img_np)
        regions = []
        
        for (bbox, text, conf) in results:
            if conf < self.min_confidence or not text.strip():
                continue
                
            # bbox 係 4 點座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x0, y0 = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x0), int(max(ys) - y0)
            
            regions.append(TextRegion(
                bbox=(x0, y0, w, h),
                text=text,
                confidence=conf,
                quad=bbox
            ))
        return regions
