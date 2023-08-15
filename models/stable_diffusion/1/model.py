#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
This module is copy-pasted in generated Triton configuration folder to perform inference.
"""

import inspect
import logging

# noinspection DuplicatedCode
from pathlib import Path
from typing import Dict, List, Union

import json

import torch
from transformers import CLIPTokenizer
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)


try:
    # noinspection PyUnresolvedReferences
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.


class TritonPythonModel:
    tokenizer: CLIPTokenizer
    device: str
    scheduler: Union[
        DDIMScheduler,
        PNDMScheduler,
        LMSDiscreteScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler,
    ]
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    eta: float

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        current_name: str = str(Path(args["model_repository"]).parent.absolute())
        self.device = "cpu" 
        if args["model_instance_kind"] == "CPU":
            self.device = "cpu"
        else 
            self.device = f"cuda:{model_instance_device_id}"
        self.tokenizer = CLIPTokenizer.from_pretrained(
            current_name + "/stable_diffusion/1/tokenizer/"
        )
        self.scheduler_config_path = current_name + "/stable_diffusion/1/scheduler/"
        self.scheduler = DPMSolverMultistepScheduler.from_config(self.scheduler_config_path)

        self.height = 512
        self.width = 512
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        self.eta = 0.0

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            prompt = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "PROMPT")
                .as_numpy()
                .tolist()
            ]
            negative_prompt = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "NEGATIVE_PROMPT")
                .as_numpy()
                .tolist()
            ]
            num_images_per_prompt = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "SAMPLES")
                .as_numpy()
                .tolist()
            ][0]
            scheduler = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "SCHEDULER")
                .as_numpy()
                .tolist()
            ][0]
            if scheduler.__class__.__name__ != scheduler:
                self.scheduler = eval(
                    f"{scheduler}.from_config(self.scheduler_config_path)"
                )
            self.num_inference_steps = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "STEPS")
                .as_numpy()
                .tolist()
            ][0]
            self.guidance_scale = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "GUIDANCE_SCALE")
                .as_numpy()
                .tolist()
            ][0]
            seed = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "SEED")
                .as_numpy()
                .tolist()
            ][0]

            ## Fix later
            if negative_prompt[0] == "NONE":
                negative_prompt = None

            # get prompt text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_input_ids = text_input.input_ids
            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                removed_text = self.tokenizer.batch_decode(
                    text_input_ids[:, self.tokenizer.model_max_length :]
                )
                logging.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

            input_ids = text_input_ids.type(dtype=torch.int32)
            inputs = [
                pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids))
            ]

            inference_request = pb_utils.InferenceRequest(
                model_name="text_encoder",
                requested_output_names=["last_hidden_state"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(
                    inference_response, "last_hidden_state"
                )
                text_embeddings: torch.Tensor = torch.from_dlpack(output.to_dlpack())

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
            text_embeddings = text_embeddings.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = self.guidance_scale > 1.0
            # get unconditional embeddings for classifier free guidance
            batch_size = 1
            if do_classifier_free_guidance:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                max_length = text_input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                input_ids = uncond_input.input_ids.type(dtype=torch.int32)
                inputs = [
                    pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids))
                ]

                inference_request = pb_utils.InferenceRequest(
                    model_name="text_encoder",
                    requested_output_names=["last_hidden_state"],
                    inputs=inputs,
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                else:
                    output = pb_utils.get_output_tensor_by_name(
                        inference_response, "last_hidden_state"
                    )
                    uncond_embeddings: torch.Tensor = torch.from_dlpack(
                        output.to_dlpack()
                    )

                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = uncond_embeddings.shape[1]
                uncond_embeddings = uncond_embeddings.repeat(
                    1, num_images_per_prompt, 1
                )
                uncond_embeddings = uncond_embeddings.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            # get the initial random noise unless the user supplied it

            latents_shape = (
                batch_size * num_images_per_prompt,
                4,
                self.height // 8,
                self.width // 8,
            )
            generator = torch.Generator(device=self.device).manual_seed(seed)
            latents = torch.randn(
                latents_shape, generator=generator, device=self.device
            )

            # set timesteps
            self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

            timesteps_tensor = self.scheduler.timesteps

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
            # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
            # and should be between [0, 1]
            accepts_eta = "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()
            )
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = self.eta

            # check if the scheduler accepts generator
            accepts_generator = "generator" in set(
                inspect.signature(self.scheduler.step).parameters.keys()
            )
            if accepts_generator:
                extra_step_kwargs["generator"] = generator

            for i, t in enumerate(timesteps_tensor):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                latent_model_input = latent_model_input.type(dtype=torch.float16)
                timestep = t[None].type(dtype=torch.float16)
                encoder_hidden_states = text_embeddings.type(dtype=torch.float16)

                inputs = [
                    pb_utils.Tensor.from_dlpack(
                        "sample", torch.to_dlpack(latent_model_input)
                    ),
                    pb_utils.Tensor.from_dlpack("timestep", torch.to_dlpack(timestep)),
                    pb_utils.Tensor.from_dlpack(
                        "encoder_hidden_states", torch.to_dlpack(encoder_hidden_states)
                    ),
                ]

                inference_request = pb_utils.InferenceRequest(
                    model_name="unet",
                    requested_output_names=["out_sample"],
                    inputs=inputs,
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                else:
                    output = pb_utils.get_output_tensor_by_name(
                        inference_response, "out_sample"
                    )
                    noise_pred: torch.Tensor = torch.from_dlpack(output.to_dlpack())

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents

            latents = latents.type(dtype=torch.float16)
            inputs = [
                pb_utils.Tensor.from_dlpack(
                    "latent_sample", torch.to_dlpack(latents)
                )
            ]
            inference_request = pb_utils.InferenceRequest(
                model_name="vae_decoder",
                requested_output_names=["sample"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(inference_response, "sample")
                image: torch.Tensor = torch.from_dlpack(output.to_dlpack())
                image = image.type(dtype=torch.float32)
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()

            tensor_output = [pb_utils.Tensor("IMAGES", image)]
            responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses
