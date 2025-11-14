import requests
import numpy as np
from PIL import Image

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((224,224))
    arr = np.array(img) / 255.0
    return arr.tolist()

img = preprocess("data/test/PNEUMONIA/person1_virus_3.jpeg")

data = {"instances": [img]}

res = requests.post("http://localhost:8501/v1/models/pneumonia:predict", json=data)
print(res.json())
