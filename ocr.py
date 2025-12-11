from transformers import AutoProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load Meta's Nougat model
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
processor = AutoProcessor.from_pretrained("facebook/nougat-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_path = 'monish_screenshot.png'

image = Image.open(image_path).convert("RGB")

# Preprocess
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Generate LaTeX/text
generated_ids = model.generate(pixel_values, max_length=1024)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("---- Recognized Text / LaTeX ----")
print(generated_text)
