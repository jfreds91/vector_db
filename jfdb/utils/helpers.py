import torch
from PIL import Image
import requests


def get_image_from_url(url: str) -> Image.Image:
  """Download and open an image from a URL.

  Args:
      url: The URL of the image to download.

  Returns:
      PIL Image object.
  """
  return Image.open(requests.get(url, stream=True).raw)


def get_embedding_from_image(image: Image.Image, processor, model) -> torch.Tensor:
  """Extract image embedding using CLIP model.

  Args:
      image: PIL Image to embed.
      processor: CLIP processor for image preprocessing.
      model: CLIP model for feature extraction.

  Returns:
      Image embedding tensor.
  """
  im_input = processor(images=image, return_tensors="pt", padding=True)
  with torch.no_grad():
    image_embeddings = model.get_image_features(**im_input)
  return image_embeddings