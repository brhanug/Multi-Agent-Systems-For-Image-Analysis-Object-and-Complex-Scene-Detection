from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import torch

model_path = "/data/brhanu/models/owlvit-base-patch16"
img_path = "/data/brhanu/thesis_project/data/colibri/images/data/PPN1752245350/00000008_0.jpg"

processor = OwlViTProcessor.from_pretrained(model_path)
model = OwlViTForObjectDetection.from_pretrained(model_path, use_safetensors=True, local_files_only=True)
model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

text = [["child", "family", "book", "horse", "tree"]]
image = Image.open(img_path).convert("RGB")
inputs = processor(text=text, images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
outputs = model(**inputs)
target_sizes = torch.Tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]

print("\nDetections:")
for s, l, b in zip(results["scores"], results["labels"], results["boxes"]):
    if s > 0.3:
        print(f"  {text[0][l]}: {s:.2f}  box={b.tolist()}")
