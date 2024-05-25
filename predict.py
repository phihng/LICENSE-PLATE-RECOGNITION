from ultralytics import YOLO
import os
import numpy as np
from PIL import Image

# Load a pretrained YOLO model
model = YOLO('E:/FIle CTU/NLN/pythonProjectNLN1/runs/detect/train2/weights/best.pt')

# Input image path
input_image_path = 'E:/FIle CTU/NLN/pythonProjectNLN1/test'

# Run inference
results = model(input_image_path)

# Specify the directory to save results
output_directory = 'E:/FIle CTU/NLN/pythonProjectNLN1/results'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Label index for license plate (you need to check the label index in your dataset)
license_plate_label_index = 0  # Replace with the correct label index

# Save images with license plate
for i, r in enumerate(results):
    for label, box in zip(r.names, np.array(r.boxes.xyxy).tolist()):
        # Check if the detected object is a license plate
        if label == license_plate_label_index:
            # Convert box coordinates to integers
            box = [int(coord) for coord in box]

            # Create image filename based on detection index
            image_filename = f'{i + 1}.png'
            # Open original image
            image_path = os.path.join(input_image_path, image_filename)
            # Check if image file exists
            if os.path.exists(image_path):
                original_image = Image.open(image_path)
                # Crop the license plate from the original image
                cropped_image = original_image.crop(box)
                # Save cropped image
                save_path = os.path.join(output_directory, f'kq_{i + 1}_bien_so.png')
                cropped_image.save(save_path)
                print(f'Cropped license plate saved to: {save_path}')
            else:
                print(f'Image file {image_filename} not found!')
            break  # Stop checking other objects in this image
