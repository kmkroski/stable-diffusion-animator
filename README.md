# Stable Diffusion Animator

This tool uses Keras' Stable Diffusion model and TensorFlow to generate animations that
help demonstrate how Stable Diffusion works. 

The tool creates two types of animations: external and internal. External animations 
show a comparison of the model's output after various amount of generational steps; 
lower values are grainy or blurry, while higher values are more detailed and more
refined. Internal animations show how the latent patch is refined to create an image 
from noise. 

For more information on how this model works, see this section of TensorFlow's docs:
https://www.tensorflow.org/tutorials/generative/generate_images_with_stable_diffusion

## Setup

This step will install SDA into a virtual environment, and then create the required 
directories and prompt YAML file.

1. Create and activate your virtual environment: 
    - `python -m venv venv`
    - `source venv/bin/activate`
2. Install SDA:
    - `pip install -e .`
3. Create and navigate to an empty project directory
    - `mkdir sda_project && cd sda_project`
4. Run SDA's `setup` command:
    - `sda setup`

## Previewing Prompts

This step generate multiple images based on the prompt. You will likely need to run this
step multiple times to "fine tune" your prompt. Once you create an image you like, you
can go on to the generation step.

1. Edit your `prompt.yml` file
    - include and exclude describe what should (or should not) be in the final image
    - adherence tells the model how "closely" it should follow the prompt. If you 
      increase this value, you'll also need to increase the number of generation steps
2. Preview your prompt:
    - `sda preview R10 16`
    - R10 is shorthand for 10 sequential seeds starting from a random number
    - 16 is the number of generation steps, 12 to 24 is a good value to start with
3. Images will saved in the `/images` folder.
    - Each image is named using the following format `{SEED}-{STEPS}.png`
    - Keep track of the seed for the next steps

If you find an image you like, but it's too blurry or grainy, re-preview it with a
larger amount of generation steps:
    - `sda preview {SEED} 24`

## Generating Frames and Animations

1. Generate the frames:
    - `sda generate {SEED} 24`
    - By default, this skips the first external frame as it looks really noisy. You can
      set the initial frame with the `-s` flag
    - The internal animation is generated along with frame that has the highest step count.
2. Generate the animation:
    - `sda animate --gif --loop`
    - This will generate two looping GIFs in the project folder.
    - One GIF shows the "external" frames: showing the output of the model over 
      different generation steps,
    - One GIF shows the "internal" frames: showing the model turning noise into an image
    - See the animation notes below for more options

## Seeds and Steps

When using the `preview` command, you can specify multiple seeds using either the random
shorthand format or a comma-separated list or both. The shorthand format is simply `R`
followed by an integer. This will create a number of sequential seeds starting from a 
random number.

For steps, you can specify a comma-separated list of values. This allows you to see the
direct difference between generational steps on the same image.

Putting this together, a command like `sda preview R5 16,24` will generate 10 images.
For each of the 5 seeds, two images will be generated: one with 16 generation steps and
one with 24 generation steps.

## Image Size Notes

The `preview` and `generate` commands allow you to specify an image size. By default, 
this tool generates images with a size of 512 by 512 pixels. You can use `-w` and `-h` 
to set the width and height respectively, but each size must be a multiple of *128*!

If you set the image dimensions too large, the model will fail during the decoding step.
On my CPU-only 16GB machine, I can reliably (but very slowly) handle images up to 768 by 768.

Note that the same seed with different image dimensions will generate _entirely_
different images.

## Animation Notes

The `animation` command has several options to customize your assembled animations:
    - `--gif` or `--mp4`: Generate a GIF or an MP4 file
    - `--loop`: Sequence the frames to create a "back-and forth" animation and 
      (GIF-only) loop the anumation. If you have 4 frames, it will sequence them as:
      1,2,3,4,3,2
    - `--fps`: Set the frames per second. If you're using a GIF, keep this low (~10), 
      but MP4s can be much higher (30 or 60 or more).
    - `--hold`: Pause on each frame for a set amount of time expressed in seconds
    - `--fade`: The duration of the fade between frames expressed in seconds
    - `--tag`: Add a small tag showing the step count. 0 will disable the tag display,
      while 1, 2, 3, or 4 will place the tag in a corner: 
        1 = top-right
        2 = top-left
        3 = bottom-left
        4 = bottom-right
    - `--internal` or `--external`: Generate either the internal or external animations,
      leaving this off will generate both.

## Example Prompt and Outputs

```
includes:
- large futuristic city settlement on Mars
- view looking down
- the city is crowded and busy and active
- gritty and lived-in
- realistic photograph
- high quality and fine details
- sharp focus and lighting
excludes:
- abstract
- drawing
- sketch
- words
- text
- watermark
adherence: 8.75
```

Preview: `sda preview R8 24`
[IMAGE]

Generate: `sda generate 4092149306 32`
Animate: 
    - `sda animate --gif --loop --hold 0.1 --fade 0.2 --tag 1 --internal`
    - `sda animate --gif --loop --hold 0.2 --fade 0.3 --external`
[IMAGES]

Generate: `sda generate 4092149312 32`
Animate: 
    - `sda animate --gif --loop --hold 0.1 --fade 0.2 --tag 1 --internal`
    - `sda animate --gif --loop --hold 0.2 --fade 0.3 --external`
[IMAGES]


## TODO

- Clean up StableDiffusionWriter and base class
- Finish tests
- Optimize result GIFs