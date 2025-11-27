import albumentations as A
import cv2
import os
import glob
from tqdm import tqdm

# ==========================================
# CONFIGURATION (Change these if needed)
# ==========================================
# Where are your original images and TXT labels?
INPUT_DIR = "raw_data"

# Where do you want the new, fake images to go?
OUTPUT_DIR = "augmented_data"

# How many new versions do you want create for EVERY original image?
# If you have 100 images and set this to 5, you will end up with 500 new images.
AUGMENTATIONS_PER_IMAGE = 5
# ==========================================


# 1. Define the "magic" transformations
# We define a list of changes the computer is allowed to make.
# We use bbox_params to ensure the boxes around food move with the image.
transform = A.Compose([
    # Randomly make the image slightly brighter or darker
    # This helps with cabin lighting variations.
    A.RandomBrightnessContrast(p=0.7),

    # Randomly flip the image horizontally (like looking in a mirror)
    # Trays can be loaded facing either direction.
    A.HorizontalFlip(p=0.5),

    # Slight rotation (plus or minus 15 degrees)
    # Trays are rarely perfectly straight.
    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),

    # Add slight "noise" (graininess) to simulate lower quality cameras
    A.ISONoise(p=0.3),

], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def define_directories():
    """Makes sure output directories exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Starting augmentation...")
    print(f"Taking images from: {INPUT_DIR}")
    print(f"Saving results to: {OUTPUT_DIR}")


def get_bboxes_from_txt(txt_path):
    """Reads the YOLO label file and gets the existing boxes."""
    bboxes = []
    labels = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # YOLO format is: class_id x_center y_center width height
                data = line.strip().split()
                class_id = int(data[0])
                bbox = [float(x) for x in data[1:]]
                
                bboxes.append(bbox)
                labels.append(class_id)
    return bboxes, labels


def save_augmented_data(image, bboxes, labels, filename_base, counter):
    """Saves the new image and its new, adjusted label file."""
    
    # New filenames (e.g., tray1_aug_0.jpg)
    new_image_name = f"{filename_base}_aug_{counter}.jpg"
    new_txt_name = f"{filename_base}_aug_{counter}.txt"
    
    image_path = os.path.join(OUTPUT_DIR, new_image_name)
    txt_path = os.path.join(OUTPUT_DIR, new_txt_name)

    # Save the image using opencv
    cv2.imwrite(image_path, image)

    # Save the new labels in YOLO format
    with open(txt_path, 'w') as f:
        for bbox, label in zip(bboxes, labels):
            # bbox points are [xc, yc, w, h]
            line = f"{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
            f.write(line)


def main():
    define_directories()
    
    # Find all .jpg files in the input directory
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    
    if not image_paths:
        print("Error: No images found in the input directory!")
        return

    # Loop through every image found (showing a progress bar)
    for img_path in tqdm(image_paths, desc="Processing Images"):
        
        # 1. Read original image
        image = cv2.imread(img_path)
        if image is None: continue
        
        # Get base filename without extension (e.g., "tray1")
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(INPUT_DIR, base_name + ".txt")

        # 2. Read original labels
        bboxes, class_labels = get_bboxes_from_txt(txt_path)

        # Skip if there are no labels for this image
        if not bboxes: continue

        # 3. Generate augmentations
        for i in range(AUGMENTATIONS_PER_IMAGE):
            try:
                # The magic happens here. Apply the transforms.
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_labels = transformed['class_labels']

                # 4. Save the result
                save_augmented_data(transformed_image, transformed_bboxes, transformed_labels, base_name, i)
            
            except Exception as e:
                print(f"Could not augment {base_name} iteration {i}. Error: {e}")

    print(f"\nDone! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()
