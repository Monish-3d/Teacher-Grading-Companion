from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Use the large handwritten TrOCR model (very accurate)
MODEL_NAME = "microsoft/trocr-large-handwritten"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", device)

# Load model and processor
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)

# Load your image
image = Image.open("monish_screenshot.png").convert("RGB")

# Process image
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Generate text
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n--- Extracted Text ---")
print(generated_text)
