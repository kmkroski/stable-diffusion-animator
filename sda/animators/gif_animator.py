from PIL.Image import Image

from .base_animator import BaseAnimator


class GIFAnimator(BaseAnimator):

    def generate(
        self,
        images: list[tuple[Image, int, int]],
        filename: str,
        tag: int = 0,
        loop: bool = False,
        fps: int = 30,
        hold_time: float = 0.5,
        fade_time: float = 1.0,
    ) -> None:
        super().generate(images, filename, tag, loop, fps, hold_time, fade_time)

        images = self._load(images, tag, loop)
        frames: list[Image] = []
        durations: list[int] = []

        for index, image in enumerate(images):
            frames.append(image)
            durations.append(hold_time * 1000)

            if not loop and image == images[-1]:
                break

            next_image = images[(index + 1) % len(images)]
            fade_images = self._fade(image, next_image, self._fade_count)

            frames.extend(fade_images)
            durations.extend([self._frame_time] * len(fade_images))

        frames[0].save(
            filename + ".gif",
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0 if loop else 1,
        )
