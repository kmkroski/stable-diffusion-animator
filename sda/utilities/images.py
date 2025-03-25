import os

from PIL import Image

WIDTH = 512
HEIGHT = 512
FORMAT = "png"

DIR_PREVIEWS = "images"
DIR_INTERNAL = "internal"
DIR_EXTERNAL = "external"


def save_image(image: Image.Image, seed: int, step: int, output_dir: str):
    """
    Save the image to disk.

    Args:
        image (Image.Image): The image to save.
        seed (int): The seed used to generate the image.
        step (int): The number of steps used to generate the image.
        output_dir (str): The directory to save the image.

    Returns:
        None
    """

    os.makedirs(output_dir, exist_ok=True)

    filename = f"{seed:04d}-{step:03d}.{FORMAT}"
    image.save(os.path.join(output_dir, filename))


def is_empty(directory: str) -> bool:
    """
    Check if a directory is empty.

    Args:
        directory (str): The directory to check.

    Returns:
        bool: True if the directory is empty, False otherwise.
    """

    if not os.path.exists(directory):
        return True

    return not bool(os.listdir(directory))


def empty_dir(directory: str) -> None:
    """
    Clear a directory of all files.

    Args:
        directory (str): The directory to clear.
    """

    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))


def _parse_image_name(name: str) -> tuple[int, int]:
    """
    Parse the seed and step from an image name.

    Args:
        name (str): The name of the image.

    Returns:
        tuple[int, int]: The seed and step parsed from the image name.
    """

    seed, step = name.split(".")[0].split("-")
    return int(seed), int(step)


def list_dir(
    directory: str, extension: str = FORMAT
) -> list[tuple[Image.Image, str, str]]:
    """
    List all files in a directory.

    Args:
        directory (str): The directory to list.
        extension (str): The file extension to filter by.

    Returns:
        list[str]: A list of all files in the directory.
    """

    files = [name for name in sorted(os.listdir(directory)) if name.endswith(extension)]

    for index, name in enumerate(files):
        seed, step = _parse_image_name(name)
        image = Image.open(os.path.join(directory, name))
        files[index] = (image, seed, step)

    return files
