import abc
import os

from PIL import Image, ImageDraw, ImageFont

FONT_FILEPATH: str = "assets/LeagueSpartan-Bold.otf"


class BaseAnimator(abc.ABC):

    _hold_count: int = 15
    _fade_count: int = 30
    _frame_time: int = round(1000 / 30)

    def __init__(self, font_file: str = FONT_FILEPATH, font_size: int = 14) -> None:
        self._font = ImageFont.truetype(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", font_file)),
            font_size,
        )

    def _load(
        self,
        images: list[tuple[Image.Image, int, int]],
        tag: int = 0,
        loop: bool = False,
    ) -> list[Image.Image]:
        images = [self._draw(image, frame, tag) for image, _, frame in images]

        if loop:
            images = images + images[-2:0:-1]

        return images

    def _draw(self, image: Image.Image, step: int, tag: int = 0) -> Image.Image:
        l, t, r, b = self._font.getbbox("00")

        step_margin = 6
        step_width = r - l + step_margin * 2
        step_height = b - t + step_margin * 2

        tag_image = Image.new("RGBA", (step_width, step_height), (45, 52, 54))
        tag_draw = ImageDraw.Draw(tag_image)
        tag_draw.text(
            (step_width / 2, step_height / 2 + 2),
            f"{step:02d}",
            font=self._font,
            fill=(223, 230, 233),
            anchor="mm",
        )

        if tag == 1:
            # top-right
            image.paste(
                tag_image,
                (image.width - step_width - step_margin, step_margin),
                tag_image,
            )

        elif tag == 2:
            # top-left
            image.paste(tag_image, (step_margin, step_margin), tag_image)

        elif tag == 3:
            # bottom-left
            image.paste(
                tag_image,
                (step_margin, image.height - step_height - step_margin),
                tag_image,
            )

        elif tag == 4:
            # bottom-right
            image.paste(
                tag_image,
                (
                    image.width - step_width - step_margin,
                    image.height - step_height - step_margin,
                ),
                tag_image,
            )

        return image

    def _fade(
        self,
        image_src: Image.Image,
        image_dst: Image.Image,
        count: int = 30,
    ) -> list[Image.Image]:
        return [Image.blend(image_src, image_dst, c / count) for c in range(count)]

    @abc.abstractmethod
    def generate(
        self,
        images: list[tuple[Image.Image, int, int]],
        filepath: str,
        tag: int = 0,
        loop: bool = False,
        fps: int = 30,
        hold_time: float = 0.5,
        fade_time: float = 1.0,
    ) -> None:
        self._hold_count = round(fps * hold_time)
        self._fade_count = round(fps * fade_time)
        self._frame_time = round(1000 / fps)

        if len(images) < 1:
            raise ValueError("No images to animate!")
