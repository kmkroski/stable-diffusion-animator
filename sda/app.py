import logging
import os

# TensorFlow and Keras are very verbose by default...
logging.basicConfig(level=logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from rich import print
from typer import Argument, Option, confirm, Abort, Typer
from typing_extensions import Annotated

from .animators.base_animator import BaseAnimator
from .animators.gif_animator import GIFAnimator
from .animators.mp4_animator import MP4Animator
from .models.helpers import (
    initialize_model,
    generate_image,
    parse_prompt,
    parse_seeds,
    parse_steps,
)
from .utilities.images import (
    WIDTH,
    HEIGHT,
    DIR_PREVIEWS,
    DIR_INTERNAL,
    DIR_EXTERNAL,
    save_image,
    is_empty,
    empty_dir,
    list_dir,
)

app = Typer(no_args_is_help=True)


def _confirm_empty(directory: str, name: str) -> None:
    if is_empty(directory):
        return

    print(
        ":warning: [bold red]Warning[/bold red]:",
        f"The {name} directory is not empty.",
    )
    delete = confirm(f"Are you sure you want to delete these files?")

    if not delete:
        raise Abort()

    empty_dir(directory)


def _generate_animation(
    animator: BaseAnimator,
    directory: str,
    suffix: str,
    tag: int,
    loop: bool,
    fps: int,
    hold_time: float,
    fade_time: float,
):
    if not is_empty(directory):
        print(
            ":robot: [bold blue]Animate[/bold blue]:",
            f"Building {suffix} animation...",
        )

        images = list_dir(directory)
        filepath = f"{images[-1][1]:010d}_{images[-1][2]:03d}_{suffix}"
        animator.generate(
            images,
            filepath,
            tag=tag,
            loop=loop,
            fps=fps,
            hold_time=hold_time,
            fade_time=fade_time,
        )


@app.command()
def setup():
    """
    Setup the environment for the Stable Diffusion model prompt and images.
    """

    parse_prompt()

    os.makedirs(DIR_PREVIEWS, exist_ok=True)
    os.makedirs(DIR_INTERNAL, exist_ok=True)
    os.makedirs(DIR_EXTERNAL, exist_ok=True)

    print(
        ":heavy_check_mark: [bold green]Success[/bold green]:",
        "Directories and prompt file setup complete!",
    )


@app.command()
def preview(
    seeds: Annotated[str, Argument(help="list of seeds")],
    steps: Annotated[str, Argument(help="generate steps")],
    width: Annotated[int, Option("--width", "-w", help="image width")] = WIDTH,
    height: Annotated[int, Option("--height", "-h", help="image height")] = HEIGHT,
) -> None:
    """
    Generate preview images from the Stable Diffusion model.
    """

    seeds = parse_seeds(seeds)
    steps = parse_steps(steps)
    include, exclude, adherence = parse_prompt()
    model = initialize_model(width, height)

    count = len(seeds) * len(steps)
    current = 0

    for seed in seeds:
        for step in steps:
            current += 1
            print(
                f":robot: [bold blue]Image {current} of {count}[/bold blue]:",
                f"Generating {width} x {height} image for seed {seed} over {step} steps",
            )

            generate_image(
                model,
                seed,
                step,
                include,
                exclude,
                adherence,
                external_callback=lambda image, seed, step: save_image(
                    image, seed, step, DIR_PREVIEWS
                ),
            )

    print(
        ":heavy_check_mark: [bold green]Success[/bold green]:",
        f"{count} preview images saved!",
    )


@app.command()
def generate(
    seed: Annotated[int, Argument(help="single seed")],
    steps: Annotated[int, Argument(help="generate steps")],
    width: Annotated[int, Option("--width", "-w", help="image width")] = WIDTH,
    height: Annotated[int, Option("--height", "-h", help="image height")] = HEIGHT,
    start: Annotated[int, Option("--start", "-s", help="start at step")] = 2,
) -> None:
    """
    Generate internal and external frames using the Stable Diffusion model.
    """

    _confirm_empty(DIR_INTERNAL, "internal frames")
    _confirm_empty(DIR_EXTERNAL, "external frames")

    include, exclude, adherence = parse_prompt()
    model = initialize_model(width, height)

    count = steps - start + 1
    current = 0

    for step in range(steps, start - 1, -1):
        current += 1
        print(
            f":robot: [bold blue]Image {current} of {count}[/bold blue]:",
            f"Generating {width} x {height} frame for seed {seed} over {step} steps",
        )

        if step == steps:
            print(
                ":robot: [bold blue]Image[/bold blue]:",
                f"Saving {step} internal frames for this external frame",
            )

        generate_image(
            model,
            seed,
            step,
            include,
            exclude,
            adherence,
            external_callback=lambda image, seed, step: (
                save_image(image, seed, step, DIR_EXTERNAL)
            ),
            internal_callback=(
                None
                if step != steps
                else lambda image, seed, step: (
                    save_image(image, seed, step, DIR_INTERNAL)
                )
            ),
        )

    print(
        ":heavy_check_mark: [bold green]Success[/bold green]:",
        f"{count} external frames and {steps} internal frames generated!",
    )


@app.command()
def animate(
    gif: Annotated[bool, Option("--gif", help="generate GIF", is_flag=True)] = False,
    mp4: Annotated[bool, Option("--mp4", help="generate MP4", is_flag=True)] = False,
    tag: Annotated[int, Option("--tag", "-t", help="tag position")] = 0,
    loop: Annotated[bool, Option("--loop", help="loop output", is_flag=True)] = False,
    fps: Annotated[int, Option("--fps", help="frames per second")] = 10,
    hold_time: Annotated[
        float, Option("--hold", help="seconds to wait on a frame")
    ] = 0.25,
    fade_time: Annotated[
        float, Option("--fade", help="seconds to fade between frames")
    ] = 0.5,
    internal: Annotated[
        bool, Option("--internal", help="generate internal animation", is_flag=True)
    ] = False,
    external: Annotated[
        bool, Option("--external", help="generate external animation", is_flag=True)
    ] = False,
):
    """
    Generate animations from the internal and external frames.
    """

    if gif and mp4:
        print(
            ":x: [bold red]Error[/bold red]:",
            "You cannot specify both --mp4 and --gif at the same time.",
            True,
        )
        raise Abort()

    if mp4 and not gif:
        animator = MP4Animator()
    elif gif and not mp4:
        animator = GIFAnimator()
    else:
        print(
            ":x: [bold red]Error[/bold red]:",
            "You must specify either --mp4 or --gif.",
        )
        raise Abort()

    if not internal and not external:
        internal = external = True

    if internal:
        _generate_animation(
            animator=animator,
            directory=DIR_INTERNAL,
            suffix="INT",
            tag=tag,
            loop=loop,
            fps=fps,
            hold_time=hold_time,
            fade_time=fade_time,
        )

    if external:
        _generate_animation(
            animator=animator,
            directory=DIR_EXTERNAL,
            suffix="EXT",
            tag=tag,
            loop=loop,
            fps=fps,
            hold_time=hold_time,
            fade_time=fade_time,
        )

    print(
        ":heavy_check_mark: [bold green]Success[/bold green]:",
        "Animations have been generated!",
    )
