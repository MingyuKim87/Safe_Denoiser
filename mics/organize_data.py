import os
import shutil


source_dir = "results/std_SD_CompVis/stable-diffusion-v1-4-i2p/unsafe"  
destination_dir = "datasets/nudity/i2p_sexual"  


os.makedirs(destination_dir, exist_ok=True)

# Copy
for filename in os.listdir(source_dir):
    if "sexual" in filename:   
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)
        shutil.copy(source_path, destination_path)
        print(f"Copied {filename} to {destination_dir}")

print("Copy finished.")