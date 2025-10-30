from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch, io

extractor = AutoFeatureExtractor.from_pretrained("Falconsai/nsfw_image_detection")
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
model.eval()

def detect_nsfw(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()
    return ["SFW", "NSFW"][predicted_class]