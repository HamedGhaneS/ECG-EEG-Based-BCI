"""
Image Resizer Script
Author: Hamed Ghane
Date: January 9, 2025

This script provides a user-friendly tool for batch resizing images based on a reference image.
It performs the following operations:
- Takes a reference image and multiple target images as input
- Displays current dimensions of all images
- Allows user to specify additional pixels for width and height
- Resizes target images to match reference dimensions plus specified offsets
- Saves resized images with '_resized' suffix
- Uses high-quality LANCZOS resampling for resizing
- Handles errors gracefully with informative messages

The script is particularly useful for standardizing image sizes while maintaining
aspect ratios, with interactive confirmation steps to prevent unwanted modifications.
"""


from PIL import Image
import os
from pathlib import Path

def show_all_image_sizes(reference_path, target_paths):
    """Show sizes of all images before proceeding"""
    print("\nCurrent Image Sizes:")
    print("-" * 50)
    
    # Show reference image size
    with Image.open(reference_path) as ref_img:
        ref_size = ref_img.size
        print(f"Reference image ({Path(reference_path).name}):")
        print(f"  Size: {ref_size[0]}x{ref_size[1]} pixels")
    
    # Show target images sizes
    print("\nImages to be resized:")
    for path in target_paths:
        with Image.open(path) as img:
            size = img.size
            print(f"\n{Path(path).name}:")
            print(f"  Size: {size[0]}x{size[1]} pixels")
    
    print("-" * 50)

def get_user_offsets():
    while True:
        try:
            print("\nHow many pixels would you like to add to the reference size?")
            width_offset = int(input("Enter additional pixels for width: "))
            height_offset = int(input("Enter additional pixels for height: "))
            return width_offset, height_offset
        except ValueError:
            print("\nPlease enter valid numbers!")

def resize_to_match_reference(reference_path, target_paths):
    # First show all image sizes
    show_all_image_sizes(reference_path, target_paths)
    
    # Get reference size
    with Image.open(reference_path) as ref_img:
        reference_size = ref_img.size
    
    # Get user input for offsets
    width_offset, height_offset = get_user_offsets()
    
    # Calculate new target size
    target_size = (reference_size[0] + width_offset, reference_size[1] + height_offset)
    print(f"\nNew target size will be: {target_size[0]}x{target_size[1]} pixels")
    
    # Confirm with user
    confirm = input("\nDo you want to proceed with these dimensions? (yes/no): ").lower()
    if confirm != 'yes':
        print("Operation cancelled.")
        return
    
    # Resize each target image
    for target_path in target_paths:
        try:
            # Open target image
            with Image.open(target_path) as img:
                # Get original size
                original_size = img.size
                print(f"\nProcessing: {Path(target_path).name}")
                print(f"Original size: {original_size[0]}x{original_size[1]} pixels")
                
                # Resize image using new target size
                resized_img = img.resize(target_size, Image.LANCZOS)
                
                # Create output filename with '_resized' suffix
                path = Path(target_path)
                new_path = path.parent / f"{path.stem}_resized{path.suffix}"
                
                # Save resized image
                resized_img.save(new_path)
                print(f"Saved resized image as: {new_path.name}")
                print(f"New size: {target_size[0]}x{target_size[1]} pixels")
                
        except Exception as e:
            print(f"Error processing {Path(target_path).name}: {e}")

# Base path
base_path = Path(r"C:\Users\stim\Documents\Hamed\ECG-EEG-BCI\2025\stimuli")

# Full paths to your images
reference_image = base_path / "Old_win.png"
new_images = [
    base_path / "win.png",
    base_path / "loss.png"
]

# Print welcome message
print("Welcome to Image Resizer!")
print("This tool will help you resize your images based on a reference image size.")
print("You can specify how many pixels to add to the reference dimensions.")

# Resize images
resize_to_match_reference(reference_image, new_images)
