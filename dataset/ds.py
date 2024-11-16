import os
import shutil
import random

# Path to your main directory
source_dir = 'C:/Users/parig/OneDrive/Sign Language Recognition/dataset'
# Directories for train, test, val
train_dir = os.path.join(source_dir, 'train')
val_dir = os.path.join(source_dir, 'val')
test_dir = os.path.join(source_dir, 'test')

# Create train, val, test directories for each alphabet
for letter in range(65, 91):  # ASCII range for A-Z
    letter_folder = chr(letter)
    
    # Create subfolders A-Z inside train, val, test
    os.makedirs(os.path.join(train_dir, letter_folder), exist_ok=True)
    os.makedirs(os.path.join(val_dir, letter_folder), exist_ok=True)
    os.makedirs(os.path.join(test_dir, letter_folder), exist_ok=True)

    # Path to current letter's images
    letter_images_path = os.path.join(source_dir, letter_folder)

    if os.path.exists(letter_images_path):
        # List all the image files
        images = os.listdir(letter_images_path)
        random.shuffle(images)  # Shuffle to randomize splitting

        # Split the images - 70% train, 15% val, 15% test
        train_images = images[:int(0.7 * len(images))]
        val_images = images[int(0.7 * len(images)):int(0.85 * len(images))]
        test_images = images[int(0.85 * len(images)):]

        # Move the images to respective folders
        for image in train_images:
            shutil.move(os.path.join(letter_images_path, image), os.path.join(train_dir, letter_folder, image))
        for image in val_images:
            shutil.move(os.path.join(letter_images_path, image), os.path.join(val_dir, letter_folder, image))
        for image in test_images:
            shutil.move(os.path.join(letter_images_path, image), os.path.join(test_dir, letter_folder, image))

print("Dataset split into train, test, and val directories.")
