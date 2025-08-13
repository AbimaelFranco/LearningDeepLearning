import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def filter_images(DATASET_PATH: str, *namefolders: str, extension: str = ".JFIF") -> int:
    """
    Deletes images that do not contain the specified extension in the given folders.

    Args:
        DATASET_PATH (str): Base path where the folders are located.
        *namefolders (str): Folder names to scan inside DATASET_PATH.
        extension (str): Byte sequence that should be present in the file header to keep the image.

    Returns:
        int: Total number of deleted images.

    Raises:
        FileNotFoundError: If any of the folders do not exist.
        OSError: If there is an error deleting files.
    """
    
    deleted_images = 0

    for folder_name in namefolders:
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if not os.path.exists(folder_path):
            print(f"Carpeta no existe: {folder_path}")
            continue

    for image in os.listdir(folder_path):
        img_path = os.path.join(folder_path, image)
        try:
            with open(img_path, "rb") as fobj:
                header_bytes = fobj.peek(10)
                is_correct = tf.compat.as_bytes(extension) in header_bytes
        except Exception as e:
            print(f"No se pudo abrir {img_path}: {e}")
            continue
        
        finally:
            fobj.close()
            
        if not is_correct:
            deleted_images += 1
            os.remove(img_path)

    return deleted_images

def images_size(DATASET_PATH: str, *namefolders: str) -> None:
    """
    Displays up to 9 images from each specified folder and shows their dimensions.

    Args:
        DATASET_PATH (str): Base directory path where folders are located.
        *namefolders (str): One or more folder names inside DATASET_PATH to process.

    Returns:
        None
    """
    for folder_name in namefolders:
        folder_path = os.path.join(DATASET_PATH, folder_name)
        images = os.listdir(folder_path)[:9]

        plt.figure(figsize=(10, 10))
        for i, image in enumerate(images):
            img_path = os.path.join(folder_path, image)
            img = mpimg.imread(img_path)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(img)
            plt.title(f"Size: {img.shape[0]} x {img.shape[1]} pixels")
            plt.axis("off")

        plt.suptitle(f"Folder: {folder_name}")
        plt.show()