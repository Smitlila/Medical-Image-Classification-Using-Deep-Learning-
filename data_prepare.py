import os
import shutil
import random
from pathlib import Path
import numpy as np

def create_directory(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15):
    """Split medical dataset into train, validation, and test sets"""
    # Create main directories
    for dir_path in [train_dir, val_dir, test_dir]:
        create_directory(dir_path)
    
    # Process each class directory
    class_dirs = [d for d in os.listdir(source_dir) 
                 if os.path.isdir(os.path.join(source_dir, d))]
    
    # Dictionary to store class statistics
    class_stats = {}
    
    for class_name in class_dirs:
        print(f"Processing {class_name} class...")
        
        # Create class directories in train, val, and test
        for dir_path in [train_dir, val_dir, test_dir]:
            create_directory(os.path.join(dir_path, class_name))
        
        # Get all medical image files
        class_dir = os.path.join(source_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
        
        if not image_files:
            print(f"No medical images found in {class_name}")
            continue
            
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split points
        n_files = len(image_files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        
        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files to respective directories
        for files, dest_dir in [
            (train_files, train_dir),
            (val_files, val_dir),
            (test_files, test_dir)
        ]:
            for f in files:
                src = os.path.join(class_dir, f)
                dst = os.path.join(dest_dir, class_name, f)
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"Error copying {f}: {str(e)}")
        
        # Store statistics
        class_stats[class_name] = {
            'train': len(train_files),
            'validation': len(val_files),
            'test': len(test_files),
            'total': n_files
        }
        
        print(f"{class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    return class_stats

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, "medical_mnist", "raw")
    train_dir = os.path.join(base_dir, "medical_mnist", "train")
    val_dir = os.path.join(base_dir, "medical_mnist", "val")
    test_dir = os.path.join(base_dir, "medical_mnist", "test")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist!")
        print("Please ensure your medical dataset is organized in the following structure:")
        print(f"{source_dir}/")
        print("├── class1/")
        print("├── class2/")
        print("└── ...(other classes)")
        return
    
    print("Starting medical dataset preparation...")
    print(f"Source directory: {source_dir}")
    
    # Split the dataset
    class_stats = split_dataset(source_dir, train_dir, val_dir, test_dir)
    
    # Print final statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    for class_name, stats in class_stats.items():
        print(f"\n{class_name}:")
        for split, count in stats.items():
            print(f"  {split}: {count} images")
    
    print("\nDataset preparation completed successfully!")
    print(f"Training data: {train_dir}")
    print(f"Validation data: {val_dir}")
    print(f"Test data: {test_dir}")

if __name__ == "__main__":
    main()