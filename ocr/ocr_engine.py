from paddleocr import PaddleOCR


class OCREngine:
    def __init__(self):
        self.model = PaddleOCR(
            lang='en',
            use_angle_cls=True
        )

    def extract(self, img_path):
        result = self.model.ocr(img_path, cls=True)

        ocr_data = []

        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            conf = line[1][1]

            ocr_data.append({
                "bbox": bbox,
                "text": text,
                "confidence": conf
            })

        return ocr_data
