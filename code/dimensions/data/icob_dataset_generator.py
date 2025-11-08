from typing import Union, Literal
import os, random
from PIL import Image, ImageDraw
from tqdm import tqdm

def create_random_monochrome_images(imgResol: Union[list, tuple] = (32, 32), 
                                    destPath: os.PathLike = "monochrome", 
                                    nImgs: int = 10000,
                                    evalComplexity: bool = True):
    """
    Create nImgs monochromatic images with random colors and specified resolution.

    Parameters:
    imgResol (tuple): A tuple containing the width and height of the images (w, h).
    destPath (str or os.PathLike): The destination path where images will be saved.
    nImgs (int): The number of images to create.
    """
    # Ensure the destination path exists
    subfolder = f"monochrome_n_{nImgs}_{imgResol[0]}x{imgResol[1]}" 
    destPath = os.path.join(destPath, subfolder)
    os.makedirs(destPath, exist_ok=True) 
    
    width, height = imgResol

    for i in tqdm(range(nImgs)):
        # Generate a random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Create a new image with the specified color
        image = Image.new("RGB", (width, height), color)
        
        # Define the file path for the new image
        file_path = os.path.join(destPath, f"monochrome_{i+1}.png")
        
        # Save the image
        image.save(file_path)

    print(f"{destPath}")


def create_chessboard_pattern_images(imgResol: Union[list, tuple] = (32, 32), 
                                     destPath: os.PathLike = os.path.curdir, 
                                     nImgs: int = 10000, 
                                     square_size: int = 8,
                                     evalComplexity: bool = True):
    """
    Create nImgs images with a chessboard pattern and specified resolution.

    Parameters:
    imgResol (tuple): A tuple containing the width and height of the images (w, h).
    destPath (str or os.PathLike): The destination path where images will be saved.
    nImgs (int): The number of images to create.
    square_size (int): The size of each square in the chessboard pattern.
    """
    # Ensure the destination path exists
    subfolder = f"chessboard_n_{nImgs}_{imgResol[0]}x{imgResol[1]}" 
    destPath = os.path.join(destPath, subfolder)
    os.makedirs(destPath, exist_ok=True) 
    
    width, height = imgResol

    for i in tqdm(range(nImgs)):
        # Generate two random colors for the chessboard pattern
        color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Create a new image with white background
        image = Image.new("RGB", (width, height), color1)
        pixels = image.load()
        
        # Draw the chessboard pattern
        for y in range(height):
            for x in range(width):
                if (x // square_size + y // square_size) % 2 == 0:
                    pixels[x, y] = color1
                else:
                    pixels[x, y] = color2
        
        # Define the file path for the new image
        file_path = os.path.join(destPath, f"chessboard_{i+1}.png")

        # Save the image
        image.save(file_path)

    print(f"{destPath}")


def create_random_shape_images(imgResol: Union[list, tuple] = (32, 32), 
                               destPath: os.PathLike = os.path.curdir, 
                               nImgs: int = 10000, 
                               shapeType: Literal["square", "circle", "triangle"] = "square", 
                               multiShape: bool = False,
                               evalComplexity: bool = True):
    """
    Create nImgs images with random basic shapes (squares, triangles, circles) and specified resolution.

    Parameters:
    imgResol (tuple): A tuple containing the width and height of the images (w, h).
    destPath (str or os.PathLike): The destination path where images will be saved.
    nImgs (int): The number of images to create.
    """
    def is_overlapping(x0, y0, x1, y1, occupied_areas):
        for area in occupied_areas:
            if not (x1 < area[0] or x0 > area[2] or y1 < area[1] or y0 > area[3]):
                return True
        return False

    # Ensure the destination path exists
    if shapeType is not None:
        subfolder = shapeType
    else:
        subfolder = ""
    if multiShape:
        subfolder = f"{subfolder}_multiShape"
    else:
        subfolder = f"{subfolder}_oneShape"
    subfolder = f"{subfolder}_n_{nImgs}_{imgResol[0]}x{imgResol[1]}" 
    destPath = os.path.join(destPath, subfolder)
    os.makedirs(destPath, exist_ok=True) 
    
    width, height = imgResol

    for i in tqdm(range(nImgs)):
        
        # Choose the background color randomly
        # background_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # image = Image.new("RGB", (width, height), background_color)
        image = Image.new("RGB", (width, height), "black")

        draw = ImageDraw.Draw(image)
        
        # Determine the number of shapes to draw (random)
        num_shapes = random.randint(1, 2) if multiShape is True else 1
        
        occupied_areas = []
        
        for _ in range(num_shapes):
            shape_type = random.choice(['square', 'triangle', 'circle']) if shapeType is None else shapeType
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            while True:
                # Prevent the infinite loop
                if len(occupied_areas) == width * height:
                    break

                elif shape_type == 'square':
                    size = random.randint(min(width, height) // 8, min(width, height) // 4)
                    x0 = random.randint(0, width - size)
                    y0 = random.randint(0, height - size)
                    x1 = x0 + size
                    y1 = y0 + size
                    if not is_overlapping(x0, y0, x1, y1, occupied_areas):
                        draw.rectangle([x0, y0, x1, y1], fill=color)
                        occupied_areas.append((x0, y0, x1, y1))
                        break

                elif shape_type == 'triangle':
                    size = random.randint(10, min(width, height) // 2)
                    x0 = random.randint(0, width - size)
                    y0 = random.randint(size, height)
                    points = [
                        (x0, y0),
                        (x0 + size, y0),
                        (x0 + size // 2, y0 - size)
                    ]
                    x1 = x0 + size
                    y1 = y0
                    x2 = x0 + size // 2
                    y2 = y0 - size
                    if not is_overlapping(min(x0, x1, x2), min(y0, y1, y2), max(x0, x1, x2), max(y0, y1, y2), occupied_areas):
                        draw.polygon(points, fill=color)
                        occupied_areas.append((min(x0, x1, x2), min(y0, y1, y2), max(x0, x1, x2), max(y0, y1, y2)))
                        break

                elif shape_type == 'circle':
                    diameter = random.randint(10, min(width, height) // 2)
                    x0 = random.randint(0, width - diameter)
                    y0 = random.randint(0, height - diameter)
                    x1 = x0 + diameter
                    y1 = y0 + diameter
                    if not is_overlapping(x0, y0, x1, y1, occupied_areas):
                        draw.ellipse([x0, y0, x1, y1], fill=color)
                        occupied_areas.append((x0, y0, x1, y1))
                        break
        
        # Define the file path for the new image
        file_path = os.path.join(destPath, f"shapes_{i+1}.png")

        image.save(file_path)

    print(f"{destPath}")


def generate_dataloader(dataDir: os.PathLike = "datasets", 
                        imgResol: tuple = (32, 32), 
                        batchSize: int = 64,
                        nImages: int = 5,
                        split: Union[list, str]=["multi_shapes", "single_shapes", "chessboards", "monochrome"]):
    
    """
    Generate a dataloader dictionary containing paths to different types of image datasets.
    Args:
        dataDir (os.PathLike, optional): The directory where the datasets will be saved. Defaults to "datasets".
        imgResol (tuple, optional): The resolution of the generated images. Defaults to (32, 32).
        batchSize (int, optional): The batch size for the dataloader. Defaults to 64.
        nImages (int, optional): The number of images to generate for each dataset type. Defaults to 5.
    Returns:
        dict: A dictionary containing paths to different types of image datasets.
            The dictionary has the following keys:
            - "root": The root directory where the datasets are saved.
            - "multi_shapes": The directory path for the dataset containing images of multiple shapes.
            - "single_shapes": The directory path for the dataset containing images of single shapes.
            - "chessboards": The directory path for the dataset containing images of chessboard patterns.
            - "monochrome": The directory path for the dataset containing monochrome images.
    """
    
    if "multi_shapes" in split:
        create_random_shape_images(imgResol=imgResol, 
                                destPath=f"{dataDir}/multi_shapes", 
                                nImgs=nImages, 
                                shapeType="square",
                                multiShape=True,
                                evalComplexity=True)
    if "single_shapes" in split:
        create_random_shape_images(imgResol=imgResol, 
                                destPath=f"{dataDir}/single_shapes", 
                                nImgs=nImages, 
                                shapeType="square",
                                multiShape=False,
                                evalComplexity=True)
    if "chessboards" in split:
        create_chessboard_pattern_images(imgResol=imgResol, 
                                        destPath=f"{dataDir}/chessboards", 
                                        nImgs=nImages,
                                        evalComplexity=True)
    if "monochrome" in split:
        create_random_monochrome_images(imgResol=imgResol, 
                                        destPath=f"{dataDir}/monochrome",
                                        nImgs=nImages,
                                        evalComplexity=True)
    
    # initialize the dictionary for the dataloader and paths
    dataset_dict = dict()
    
    # Save the paths to the images in separate keys
    dataset_dict["root"] = dataDir
    dataset_dict["multi_shapes"] = f"{dataDir}/multi_shapes"
    dataset_dict["single_shapes"] = f"{dataDir}/single_shapes"
    dataset_dict["chessboards"] = f"{dataDir}/chessboards"
    dataset_dict["monochrome"] = f"{dataDir}/monochrome"

    return dataset_dict 
