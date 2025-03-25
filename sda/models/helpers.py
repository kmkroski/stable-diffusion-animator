import os
import random
import yaml

from .stable_diffusion import StableDiffusionWriter

PROMPT_FILENAME: str = "prompt.yml"
PROMPT_GLUE: str = ". "
PROMPT_INCLUDES: list[str] = [
    "realistic photograph",
    "high quality and fine details",
    "sharp focus and lighting",
]
PROMPT_EXCLUDES: list[str] = [
    "blue sky",
    "abstract",
    "drawing",
    "sketch",
    "words",
    "text",
    "watermark",
]
PROMPT_ADHERENCE: float = 8.25


def _seed_value(value: str) -> list[int]:
    value = value.strip().upper()

    if value.isnumeric() and int(value) > 0:
        return [int(value)]
    if value.startswith("R") and value[1:].isnumeric():
        seed_count = int(value[1:])
        base_seed = random.randint(0, 2**32 - 1)
        return [base_seed + i for i in range(seed_count)]

    raise ValueError(f"Invalid seed value: {value}")


def parse_seeds(seeds: str) -> list[int]:
    """
    Parses the seeds string into a list of integers.

    Args:
        seeds (str): The seeds string to parse.

    Returns:
        list[int]: A list of integers parsed from the seeds string.
    """

    if not seeds.strip():
        raise ValueError("Seed value cannot be empty.")

    seed_lists = [_seed_value(seed) for seed in seeds.split(",")]
    return [seed for seed_group in seed_lists for seed in seed_group]


def parse_steps(value: str) -> list[int]:
    """
    Parse a comma-separated string of step values into a list of integers.

    Args:
        value (str): A comma-separated string of step values.

    Returns:
        list[int]: A list of integers representing the steps.

    Raises:
        ValueError: If the step value is empty or invalid.
    """

    if not value.strip():
        raise ValueError("Step value cannot be empty.")

    return [int(step.strip()) for step in value.split(",")]


def parse_prompt(
    filename: str = PROMPT_FILENAME,
    glue: str = PROMPT_GLUE,
) -> tuple[str, str, float]:
    """
    Parse the prompt configuration from a YAML file.

    Args:
        filename (str): The path to the YAML file containing the prompt configuration.
        glue (str): The string used to join the prompt includes and excludes.

    Returns:
        tuple[str, str, float]: A tuple containing the includes, excludes, and adherence.

    Raises:
        ValueError: If the YAML file contains invalid data.
    """

    prompt = {
        "includes": PROMPT_INCLUDES,
        "excludes": PROMPT_EXCLUDES,
        "adherence": PROMPT_ADHERENCE,
    }

    if not os.path.exists(filename):
        with open(filename, "w") as file:
            yaml.dump(prompt, file, sort_keys=False)
    else:
        with open(filename, "r") as file:
            prompt.update(yaml.safe_load(file))

    if not isinstance(prompt["includes"], list) or not isinstance(
        prompt["excludes"], list
    ):
        raise ValueError(
            "Invalid prompt format: 'includes' and 'excludes' must be lists."
        )
    if not isinstance(prompt["adherence"], (int, float)):
        raise ValueError("Invalid prompt format: 'adherence' must be a number.")

    return (
        glue.join(prompt["includes"]),
        glue.join(prompt["excludes"]),
        float(prompt["adherence"]),
    )


def initialize_model(
    image_width: int,
    image_height: int,
) -> StableDiffusionWriter:
    """
    Initializes and returns a StableDiffusionWriter instance with the specified image dimensions.

    Args:
        image_width (int): The width of the image in pixels. Defaults to IMAGE_WIDTH.
        image_height (int): The height of the image in pixels. Defaults to IMAGE_HEIGHT.

    Returns:
        StableDiffusionWriter: An instance of StableDiffusionWriter configured with the given dimensions.
    """

    return StableDiffusionWriter(
        img_width=image_width,
        img_height=image_height,
    )


def generate_image(
    model: StableDiffusionWriter,
    seed: int,
    steps: int,
    include: str,
    exclude: str,
    adherence: float,
    internal_callback: any = None,
    external_callback: any = None,
) -> None:
    """
    Generates an image using the Stable Diffusion model based on the provided prompts and parameters.

    Args:
        model (StableDiffusionWriter): The Stable Diffusion model instance used for image generation.
        seed (int): The random seed for reproducibility of the generated image.
        steps (int): The number of diffusion steps to perform during image generation.
        include (str): The text prompt to guide the image generation process (positive prompt).
        exclude (str): The text prompt to avoid during the image generation process (negative prompt).
        adherence (float): The guidance scale for controlling adherence to the prompts.
        internal_callback (any, optional): A callback function to handle the image after each diffusion step. Defaults to None.
        external_callback (any, optional): A callback function to handle the image after all diffusion steps. Defaults to None.

    Returns:
        None: This function does not return a value. Use the callbacks to obtain the generated image(s).
    """

    model.text_to_image(
        include_prompt=include,
        exclude_prompt=exclude,
        seed=seed,
        num_steps=steps,
        unconditional_guidance_scale=adherence,
        internal_callback=internal_callback,
        external_callback=external_callback,
    )
