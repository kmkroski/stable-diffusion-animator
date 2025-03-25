import math
import numpy as np

from keras_cv.src.backend import ops
from keras_cv.src.models.stable_diffusion.stable_diffusion import StableDiffusion
from tensorflow import keras
from PIL import Image


class StableDiffusionWriter(StableDiffusion):

    def text_to_image(
        self,
        include_prompt="",
        exclude_prompt="",
        num_steps=50,
        unconditional_guidance_scale=7.5,
        seed=None,
        external_callback=None,
        internal_callback=None,
    ):
        return self.generate_image(
            include_prompt=include_prompt,
            exclude_prompt=exclude_prompt,
            num_steps=num_steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            seed=seed,
            external_cb=external_callback,
            internal_cb=internal_callback,
        )

    def generate_image(
        self,
        include_prompt,
        exclude_prompt=None,
        batch_size=1,
        num_steps=50,
        unconditional_guidance_scale=7.5,
        diffusion_noise=None,
        seed=None,
        external_cb=None,
        internal_cb=None,
    ):
        if diffusion_noise is not None and seed is not None:
            raise ValueError(
                "`diffusion_noise` and `seed` should not both be passed to "
                "`generate_image`. `seed` is only used to generate diffusion "
                "noise when it's not already user-specified."
            )

        encoded_text = self.encode_text(include_prompt)
        context = self._expand_tensor(encoded_text, batch_size)

        unconditional_text = self.encode_text(exclude_prompt)
        unconditional_context = self._expand_tensor(unconditional_text, batch_size)

        latent = self._get_initial_diffusion_noise(batch_size, seed)

        # Iterative reverse diffusion stage
        num_timesteps = 1000
        ratio = (
            (num_timesteps - 1) / (num_steps - 1) if num_steps > 1 else num_timesteps
        )
        timesteps = (np.arange(0, num_steps) * ratio).round().astype(np.int64)

        alphas, alphas_prev = self._get_initial_alphas(timesteps)
        progbar = keras.utils.Progbar(len(timesteps))
        iteration = 0
        for index, timestep in list(enumerate(timesteps))[::-1]:
            latent_prev = latent  # Set aside the previous latent vector
            t_emb = self._get_timestep_embedding(timestep, batch_size)
            unconditional_latent = self.diffusion_model.predict_on_batch(
                {
                    "latent": latent,
                    "timestep_embedding": t_emb,
                    "context": unconditional_context,
                }
            )
            latent = self.diffusion_model.predict_on_batch(
                {
                    "latent": latent,
                    "timestep_embedding": t_emb,
                    "context": context,
                }
            )
            latent = ops.array(
                unconditional_latent
                + unconditional_guidance_scale * (latent - unconditional_latent)
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            # Keras backend array need to cast explicitly
            target_dtype = latent_prev.dtype
            latent = ops.cast(latent, target_dtype)
            pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(a_t)
            latent = (
                ops.array(latent) * math.sqrt(1.0 - a_prev)
                + math.sqrt(a_prev) * pred_x0
            )

            iteration += 1
            progbar.update(iteration)

            if internal_cb is not None:
                internal_cb(self.decode_image(latent), seed, iteration)

        image = self.decode_image(latent)
        if external_cb is not None:
            external_cb(image, seed, iteration)

        return image

    def decode_image(self, latent) -> Image.Image:
        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        return Image.fromarray(np.clip(decoded, 0, 255).astype("uint8")[0])
