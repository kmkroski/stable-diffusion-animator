from cv2 import (
    VideoWriter,
    VideoWriter_fourcc,
    cvtColor,
    destroyAllWindows,
    COLOR_RGB2BGR,
)
from numpy import array as np_array
from PIL.Image import Image

from .base_animator import BaseAnimator


class MP4Animator(BaseAnimator):

    def _convert(self, image: Image) -> any:
        return cvtColor(np_array(image), COLOR_RGB2BGR)

    def generate(
        self,
        images: list[tuple[Image, int, int]],
        filepath: str,
        tag: int = 0,
        loop: bool = False,
        fps: int = 30,
        hold_time: float = 0.5,
        fade_time: float = 1.0,
    ) -> None:
        super().generate(images, filepath, tag, loop, fps, hold_time, fade_time)

        images = self._load(images, tag, loop)
        video = VideoWriter(
            filepath,
            VideoWriter_fourcc(*"mp4v"),
            fps,
            images[0].size,
        )

        for index, image in enumerate(images):
            for _ in range(self._hold_count):
                video.write(self._convert(image))

            if not loop and index == len(images) - 1:
                break

            next_image = images[(index + 1) % len(images)]
            for fade_frame in self._fade(image, next_image, self._fade_count):
                video.write(self._convert(fade_frame))

        destroyAllWindows()
        video.release()
