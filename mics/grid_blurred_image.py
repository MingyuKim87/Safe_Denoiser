import os
import random
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF

def create_blurred_grid_image(folder_path, output_pdf, grid_size=(5, 15), blur_ksize=(151, 151), quality=80):
    # Get all PNG files in the folder
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    if len(png_files) < grid_size[0] * grid_size[1]:
        raise ValueError("Not enough images in the folder to create the grid.")
    
    # Randomly select 75 images
    selected_files = random.sample(png_files, grid_size[0] * grid_size[1])
    
    # Load and process images
    images = []
    for file in selected_files:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        img = cv2.GaussianBlur(img, blur_ksize, 0)  # Apply maximum Gaussian Blur
        images.append(img)
    
    # Determine individual image size
    img_h, img_w, _ = images[0].shape
    
    # Create a blank canvas for the grid
    grid_h = img_h * grid_size[0]
    grid_w = img_w * grid_size[1]
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Place images on the grid
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y_offset = i * img_h
            x_offset = j * img_w
            grid_image[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = images[i * grid_size[1] + j]
    
    # Convert to PIL image and save as compressed PNG
    grid_pil = Image.fromarray(grid_image)
    temp_img_path = "temp_grid_image.jpg"
    grid_pil.save(temp_img_path, format="JPEG", quality=quality, optimize=True)  # Compress image
    
    # Create PDF
    pdf = FPDF(unit="pt", format=[grid_w, grid_h])
    pdf.add_page()
    pdf.image(temp_img_path, 0, 0, grid_w, grid_h)
    pdf.output(output_pdf)
    os.remove(temp_img_path) 
    
def create_blurred_grid_image_for_different_size(folder_path, output_pdf, grid_size=(20, 15), blur_ksize=(151, 151), quality=80, target_size=(200, 200)):
    # Get all PNG files in the folder
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    if len(png_files) < grid_size[0] * grid_size[1]:
        raise ValueError("Not enough images in the folder to create the grid.")
    
    # Randomly select images
    selected_files = random.sample(png_files, grid_size[0] * grid_size[1])
    
    # Load and process images
    images = []
    for file in selected_files:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        # img = cv2.GaussianBlur(img, blur_ksize, 0)  # Apply maximum Gaussian Blur
        img = cv2.resize(img, target_size)  # Resize to uniform size
        images.append(img)
    
    # Determine individual image size
    img_h, img_w = target_size
    
    # Create a blank canvas for the grid
    grid_h = img_h * grid_size[0]
    grid_w = img_w * grid_size[1]
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Place images on the grid
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y_offset = i * img_h
            x_offset = j * img_w
            grid_image[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = images[i * grid_size[1] + j]
    
    # Convert to PIL image and save as compressed JPEG
    grid_pil = Image.fromarray(grid_image)
    temp_img_path = "temp_grid_image.jpg"
    grid_pil.save(temp_img_path, format="JPEG", quality=quality, optimize=True)  # Compress image
    
    # Create PDF
    pdf = FPDF(unit="pt", format=[grid_w, grid_h])
    pdf.add_page()
    pdf.image(temp_img_path, 0, 0, grid_w, grid_h)
    pdf.output(output_pdf)
    os.remove(temp_img_path) 

# Example usage
folder_path = "results/coco30k/negation_4/safree_neg_prompt_rep_threshold_time_SD_v3v1-4_coco30k/all"  
output_pdf = "ref_imgs_ours_coco30k.pdf"  # 저장할 PDF 파일명
create_blurred_grid_image_for_different_size(folder_path, output_pdf)
