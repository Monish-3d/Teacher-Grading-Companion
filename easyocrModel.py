import easyocr

reader = easyocr.Reader(['en'], gpu=True)

result = reader.readtext('monish_screenshot.png')

# Print the recognized text.
for (bbox, text, prob) in result:
    print(f"Detected text: {text} (Confidence: {prob:.2f})")


