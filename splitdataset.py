import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
source_dir = "SignImage48x48"
output_dir = "DatasetSplit"  # Output directory for train/test/val

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create directories for train, val, and test
splits = ["train", "val", "test"]
for split in splits:
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        os.makedirs(os.path.join(output_dir, split, letter), exist_ok=True)

# Iterate through each class (A-Z)
for letter in os.listdir(source_dir):
    class_dir = os.path.join(source_dir, letter)
    if os.path.isdir(class_dir):  # Ensure it's a directory
        images = os.listdir(class_dir)
        images = [img for img in images if img.endswith((".png", ".jpg", ".jpeg"))]

        # Split the images
        train_images, temp_images = train_test_split(images, test_size=val_ratio + test_ratio, random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

        # Copy images to respective directories
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, "train", letter))
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, "val", letter))
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, "test", letter))

print("Dataset split completed!")
