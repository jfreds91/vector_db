import torch
from PIL import Image
import requests

def get_image_from_url(url:str):
  return Image.open(requests.get(url, stream=True).raw)

def get_embedding_from_image(image:Image.Image, processor, model):
  im_input = processor(images=image, return_tensors="pt", padding=True)
  with torch.no_grad():
    image_embeddings = model.get_image_features(**im_input)
  return image_embeddings