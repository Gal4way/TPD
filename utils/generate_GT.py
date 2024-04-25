import os
from PIL import Image
from torchvision.transforms import ToTensor

def get_tensor():
    return ToTensor()

def resize_images(source_dir, target_dir, target_size):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)

            source_img = Image.open(source_path).convert("RGB")
            source_img = source_img.resize(target_size, Image.BILINEAR)
            source_img.save(target_path)

    print("Image resizing completed.")

source_directory = "../datasets/test/image"
target_directory = "../datasets/test/image_512"

target_size = (384, 512) 

resize_images(source_directory, target_directory, target_size)
