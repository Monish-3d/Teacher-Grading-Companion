"""
handwriting_ocr_large.py
Usage:
    python handwriting_ocr_large.py

Requirements:
    pip install transformers pillow opencv-python torch torchvision --index-url https://download.pytorch.org/whl/cu126
(Use the cu126 wheel if you have a CUDA 12.6-capable setup; otherwise install the appropriate torch wheel.)
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import os

# --------- CONFIG ----------
IMAGE_PATH = "monish_screenshot.png"
OUTPUT_TEXT = "recognized_handwriting_large.txt"
OUTPUT_PREVIEW = "preview_lines_large.png"
MODEL_NAME = "microsoft/trocr-large-handwritten"   # the stronger model
BATCH_SIZE = 4            # how many line crops to infer at once
DRAW_PREVIEW = True       # draw bounding boxes and save preview image
# --------------------------

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_model_and_processor(model_name=MODEL_NAME, device=None):
    print(f"[INFO] Loading model {model_name} ...")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    if device is not None:
        model.to(device)
    model.eval()
    return processor, model

def segment_lines_opencv(img_gray):
    """
    Returns list of bounding boxes (x,y,w,h) sorted top->bottom for line regions.
    """
    # denoise
    denoised = cv2.fastNlMeansDenoising(img_gray, h=15)
    # adaptive threshold (inverse for white background -> black text)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 15)

    # morphological ops to merge characters into line blobs
    # kernel width should be large to join words on the same line; adjust if needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (240, 7))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # filter tiny boxes (noise)
        if w > 100 and h > 20:
            boxes.append((x, y, w, h))

    # sort top to bottom
    boxes = sorted(boxes, key=lambda b: b[1])
    return boxes, dilated

def crop_line_image(gray, box, pad=10):
    x, y, w, h = box
    h_img, w_img = gray.shape
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    crop = gray[y1:y2, x1:x2]
    # convert single-channel to RGB image for processor
    pil = Image.fromarray(crop).convert("RGB")
    pil = ImageOps.autocontrast(pil)
    # optionally resize very narrow/wide crops to a reasonable width for model
    max_width = 1600
    if pil.width > max_width:
        wpercent = max_width / float(pil.size[0])
        hsize = int(float(pil.size[1]) * float(wpercent))
        pil = pil.resize((max_width, hsize), Image.BICUBIC)
    return pil

def infer_lines(processor, model, device, line_images):
    """
    line_images: list of PIL images
    returns list of decoded strings (in the same order)
    """
    texts = []
    # Process in batches
    for i in range(0, len(line_images), BATCH_SIZE):
        batch = line_images[i:i+BATCH_SIZE]
        pixel_values = processor(images=batch, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        # generate
        with torch.no_grad():
            generated_ids = model.generate(pixel_values,
                                           max_new_tokens=256,
                                           num_beams=4,
                                           length_penalty=1.0,
                                           early_stopping=True)
        # decode on CPU to avoid GPU->CPU overhead in text ops
        generated_ids = generated_ids.cpu()
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        # strip whitespace
        decoded = [d.strip() for d in decoded]
        texts.extend(decoded)
    return texts

def main():
    device = get_device()
    print(f"[INFO] Device: {device}")

    processor, model = load_model_and_processor(MODEL_NAME, device=device)

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # read image
    orig = cv2.imread(IMAGE_PATH)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    boxes, dilated = segment_lines_opencv(gray)
    print(f"[INFO] Detected {len(boxes)} line boxes")

    line_images = []
    boxes_kept = []
    for box in boxes:
        pil = crop_line_image(gray, box, pad=12)
        # skip lines that are almost blank
        bw = pil.convert("L")
        arr = np.array(bw)
        if arr.mean() < 250:  # mostly not blank (tunable)
            line_images.append(pil)
            boxes_kept.append(box)

    if len(line_images) == 0:
        print("[WARN] No lines found after filtering. Try adjusting kernel or thresholds.")
        return

    print(f"[INFO] Running TrOCR on {len(line_images)} lines (batch_size={BATCH_SIZE}) ...")
    recognized = infer_lines(processor, model, device, line_images)

    # Combine results
    final_text = "\n".join(recognized)
    with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"[INFO] Recognized text saved to: {OUTPUT_TEXT}")

    # Optionally draw preview image
    if DRAW_PREVIEW:
        preview = orig.copy()
        for (x, y, w, h), txt in zip(boxes_kept, recognized):
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # put a small index label
            cv2.putText(preview, str(recognized.index(txt)+1), (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imwrite(OUTPUT_PREVIEW, preview)
        print(f"[INFO] Preview image saved to: {OUTPUT_PREVIEW}")

    print("\n========== RECOGNIZED TEXT ==========\n")
    print(final_text)
    print("\n=====================================\n")

if __name__ == "__main__":
    main()
