import os
import pandas as pd
from datasets import load_dataset
import random

if __name__ == "__main__":
    image_save_path = "test_images"
    os.makedirs(image_save_path, exist_ok=True)
    dataset = load_dataset("UCSC-VLAA/Recap-COCO-30K", split="train")
    
    # pick image by iamge_id
    target_image_id = "459"
    filtered_data = [sample for sample in dataset if sample["image_id"] == target_image_id]
    filtered_image = filtered_data['image']
    image_file_path = os.path.join(image_save_path, f"image_{target_image_id}.jpg")
    filtered_image.save(image_file_path)
    
    '''# loop
    for idx, sample in enumerate(dataset):
        image = sample["image"]
        image_id = sample["image_id"]
        
        image_file_path = os.path.join(image_save_path, f"image_{image_id}.jpg")
        image.save(image_file_path)
    '''