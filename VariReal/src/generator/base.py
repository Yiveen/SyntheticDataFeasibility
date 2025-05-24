from abc import ABC, abstractmethod
import random

import numpy as np
import torch
from diffusers import DiffusionPipeline, AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoPipelineForText2Image, DPMSolverMultistepScheduler

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
)

from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from typing import List

class Sampler:
    """
    Prompt sampler for generating a fixed number of prompts per real image.

    Attributes:
        images_per_real (int): Number of prompts to sample.
        feasibility (bool): Whether to use feasible or infeasible prompt list.
        previous_indices (List[List[str]]): Stores last 5 used prompt sets to avoid duplicates.
    """
    def __init__(self, images_per_real, feasibility):
        self.images_per_real = images_per_real
        self.feasibility = feasibility
        self.previous_indices = []

    def sample_words(self, choosed_feasible_list, choosed_infeasible_list):
        """
        Samples a list of prompts.

        If required number exceeds available, uses sampling with replacement.
        Keeps track of previously sampled combinations to avoid duplicates.

        Returns:
            List[str]: List of sampled prompt words.
        """
        # random.seed(3407) 

        sample_list = choosed_feasible_list if self.feasibility else choosed_infeasible_list
        total_count = len(sample_list)

        if self.images_per_real <= total_count:
            while True:
                prompt_words = random.sample(sample_list, self.images_per_real)
                if prompt_words not in self.previous_indices:
                    self.previous_indices.append(prompt_words)
                    if len(self.previous_indices) > 5:
                        self.previous_indices.pop(0)
                    break
        else:
            prompt_words = sample_list[:]  
            additional_samples_needed = self.images_per_real - total_count
            
            if additional_samples_needed > 0:
                prompt_words += random.choices(sample_list, k=additional_samples_needed)
        
        return prompt_words


class BaseGenerator(ABC):
    """
    Abstract base class for diffusion-based image generators with support for:
    - Feasible/infeasible prompt control
    - SDXL / ControlNet / SD2 pipelines
    - Text2Image / Inpainting / Image2Image generation

    Subclasses must implement preprocessing, generation, and prompt logic.
    """
    CATEGORY_MAP = {
                '707-320': 'civilian',
                '727-200': 'civilian',
                '737-200': 'civilian',
                '737-300': 'civilian',
                '737-400': 'civilian',
                '737-500': 'civilian',
                '737-600': 'civilian',
                '737-700': 'civilian',
                '737-800': 'civilian',
                '737-900': 'civilian',
                '747-100': 'civilian',
                '747-200': 'civilian',
                '747-300': 'civilian',
                '747-400': 'civilian',
                '757-200': 'civilian',
                '757-300': 'civilian',
                '767-200': 'civilian',
                '767-300': 'civilian',
                '767-400': 'civilian',
                '777-200': 'civilian',
                '777-300': 'civilian',
                'A300B4': 'civilian',
                'A310': 'civilian',
                'A318': 'civilian',
                'A319': 'civilian',
                'A320': 'civilian',
                'A321': 'civilian',
                'A330-200': 'civilian',
                'A330-300': 'civilian',
                'A340-200': 'civilian',
                'A340-300': 'civilian',
                'A340-500': 'civilian',
                'A340-600': 'civilian',
                'A380': 'civilian',
                'ATR-42': 'civilian',
                'ATR-72': 'civilian',
                'An-12': 'military',
                'BAE 146-200': 'civilian',
                'BAE 146-300': 'civilian',
                'BAE-125': 'civilian',
                'Beechcraft 1900': 'civilian',
                'Boeing 717': 'civilian',
                'C-130': 'military',
                'C-47': 'military',
                'CRJ-200': 'civilian',
                'CRJ-700': 'civilian',
                'CRJ-900': 'civilian',
                'Cessna 172': 'civilian',
                'Cessna 208': 'civilian',
                'Cessna 525': 'civilian',
                'Cessna 560': 'civilian',
                'Challenger 600': 'civilian',
                'DC-10': 'civilian',
                'DC-3': 'civilian',
                'DC-6': 'civilian',
                'DC-8': 'civilian',
                'DC-9-30': 'civilian',
                'DH-82': 'military',
                'DHC-1': 'military',
                'DHC-6': 'civilian',
                'DHC-8-100': 'civilian',
                'DHC-8-300': 'civilian',
                'DR-400': 'civilian',
                'Dornier 328': 'civilian',
                'E-170': 'civilian',
                'E-190': 'civilian',
                'E-195': 'civilian',
                'EMB-120': 'civilian',
                'ERJ 135': 'civilian',
                'ERJ 145': 'civilian',
                'Embraer Legacy 600': 'civilian',
                'Eurofighter Typhoon': 'military',
                'F-16A/B': 'military',
                'F/A-18': 'military',
                'Falcon 2000': 'civilian',
                'Falcon 900': 'civilian',
                'Fokker 100': 'civilian',
                'Fokker 50': 'civilian',
                'Fokker 70': 'civilian',
                'Global Express': 'civilian',
                'Gulfstream IV': 'civilian',
                'Gulfstream V': 'civilian',
                'Hawk T1': 'military',
                'Il-76': 'military',
                'L-1011': 'civilian',
                'MD-11': 'civilian',
                'MD-80': 'civilian',
                'MD-87': 'civilian',
                'MD-90': 'civilian',
                'Metroliner': 'civilian',
                'Model B200': 'civilian',
                'PA-28': 'civilian',
                'SR-20': 'civilian',
                'Saab 2000': 'civilian',
                'Saab 340': 'civilian',
                'Spitfire': 'military',
                'Tornado': 'military',
                'Tu-134': 'civilian',
                'Tu-154': 'civilian',
                'Yak-42': 'civilian'
            }
    def __init__(self, logger=None, use_mask=True, with_image=False,
                 feasibility=False, cache_dir=None, use_compel=True, 
                 feasible_prompts=None, infeasible_prompts=None, images_per_real=5, batch_size=1, general_class=None,
                 dataset=None, give_post=False,category=None):
        '''
        Args:
            image:  BGR rank, for cv2 process,
            mask:
            seed:
            logger:
            use_mask:
            with_image:
            feasibility:
            cache_dir:
            use_compel:
            feasible_prompts:
            infeasible_prompts:
            images_per_real:
            batch_size:
            general_class:
            cur_class:
            dataset:
        '''
        self.image = None
        self.image_bgr = None
        self.mask = None
        
        self.logger = logger
        self.feasibility = feasibility
        self.use_mask = use_mask
        self.with_image = with_image
        self.use_compel = use_compel

        self.images_per_real = images_per_real
        self.batch_size = batch_size
        if not self.use_mask:
            self.batch_size = 1 # for baseline settings, we just give bs=1

        self.dataset = dataset
        self.general_class = general_class

        self.feasible_prompts = feasible_prompts 
        self.infeasible_prompts = infeasible_prompts

        self.cache_dir = cache_dir
        self.config = None

        # if seed is None:
        #     self.seed = random.randint(0, 2 ** 32 - 1)
        # else:
        #     self.seed = seed

        # self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.pipe = None
        self.category = category
        
        self.give_post = give_post
        
    def initialize_pipeline(self, image, mask, cur_class):
        """
        Setup initial image, mask, and generate base prompts.
        """
        self.image = image
        self.image_bgr = image[:, :, ::-1]
        self.mask = np.where(mask < 128, 0, 255).astype(np.uint8)
        
        if self.feasible_prompts is not None and self.infeasible_prompts is not None:
            if self.dataset == 'fgvc_aircraft' and self.category == 'texture':
                try:
                    cur_class_catgory = self.CATEGORY_MAP[cur_class]
                except:
                    cur_class_catgory = cur_class    
            else:
                cur_class_catgory = cur_class
            self.prompt_words = self.get_base_prompt(self.dataset, cur_class_catgory)

    def _get_generator(self):
        """
        Create a seeded generator for reproducible results.
        """
        seed_list = [42, 3407]
        random_seed = random.choice(seed_list)
        self.logger.log(f"### Current random seed is {random_seed} ###")
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        return generator

    @abstractmethod
    def pre_process(self, preprocess_prompts=None):
        """pre process the input image and mask"""
        pass

    @abstractmethod
    def generate(self, input_prompts, input_images, mask_images, other_process_dict):
        """method to use pipeline to generate images"""
        pass
    
    @abstractmethod
    def pre_process_batched(self, preprocess_prompts=None):
        """pre process the input image and mask"""
        pass

    @abstractmethod
    def generate_batched(self, input_prompts, input_images, mask_images, other_process_dict):
        """method to use pipeline to generate images"""
        pass
    
    @abstractmethod
    def get_prompt_batched(self, cur_class):
        """get the current generation prompt"""
        pass

    @abstractmethod
    def _load_pipeline_model(self):
        """load corresponding pipeline"""
        pass

    def load_stable_diffision2(self):
        """
        Load SD2.1 base pipeline with CPU offloading and Compel.
        """
        self.logger.log("### Loading the Stable Diffusion 2 ###")
        repo_id = "stabilityai/stable-diffusion-2-1-base"
        pipe_sd2 = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16",
                                                     cache_dir=self.cache_dir)

        pipe_sd2.scheduler = DPMSolverMultistepScheduler.from_config(pipe_sd2.scheduler.config)
        pipe_sd2.enable_model_cpu_offload()
        # Check if PyTorch version is 2.x
        # if int(torch.__version__.split(".")[0]) >= 2:
        #     pipe_sd2.unet = torch.compile(pipe_sd2.unet, mode="reduce-overhead", fullgraph=True)
        self.logger.log("### Finish loading ###")

        if self.use_compel:
            compel_sd2 = Compel(tokenizer=pipe_sd2.tokenizer, text_encoder=pipe_sd2.text_encoder)
            return pipe_sd2, compel_sd2
        else:
            return pipe_sd2

    def load_sdxl_model(self):
        """
        Load SDXL pipeline for inpainting or text2img (optionally img2img).
        """
        if self.use_mask:
            self.logger.log("### Loading the SDXL inpainting ###")
            pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                             torch_dtype=torch.float16, variant="fp16",
                                                             cache_dir=self.cache_dir)
            # Check if PyTorch version is 2.x
            # if int(torch.__version__.split(".")[0]) >= 2:
            #     pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

            pipe.enable_model_cpu_offload()
            self.logger.log("### Finish loading ###")

            if self.use_compel:
                compel = Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True]
                )

                return pipe, compel
            else:
                return pipe
        else:
            # in this setting, we force to set not to use compel
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
                use_safetensors=True, cache_dir=self.cache_dir,
            )
            if self.with_image:
                pipe = AutoPipelineForImage2Image.from_pipe(pipe)
            pipe.enable_model_cpu_offload()
            return pipe

    def load_controlnet_model(self):
        """
        Load SDXL ControlNet pipeline with optional IP-Adapter and Compel.
        """
        self.logger.log("### Loading the SDXL ControlNet Models ###")

        # Load ControlNet models for Canny and Depth
        controlnet_canny = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )

        # controlnet_depth = ControlNetModel.from_pretrained(
        #     "diffusers/controlnet-depth-sdxl-1.0",
        #     torch_dtype=torch.float16,
        #     cache_dir=self.cache_dir
        # )

        # Load the VAE model
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )

        # Initialize the pipeline with both ControlNet models
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0",
            torch_dtype=torch.float16,
            variant="fp16",
            controlnet=controlnet_canny,
            vae=vae,
            cache_dir=self.cache_dir
        )
        
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name=[
                "ip-adapter-plus_sdxl_vit-h.safetensors",
            ],
            image_encoder_folder="models/image_encoder",
            cache_dir=self.cache_dir,
        )
        if self.dataset == 'oxford_pets':
            pipe.set_ip_adapter_scale(0.7)
        elif self.dataset == 'cars':
            pipe.set_ip_adapter_scale(0.4)
        else:
            pipe.set_ip_adapter_scale(0.4)

        pipe.enable_model_cpu_offload()
        self.logger.log("### Finish loading ###")

        if self.use_compel:

            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            # compel = Compel(
            #     tokenizer=pipe.tokenizer,  # single tokenizer
            #     text_encoder=pipe.text_encoder,  # single text encoder
# )

            return pipe, compel
        else:
            return pipe
        

    def get_base_prompt(self, dataset, _class):
        """
        Fetch sampled prompts for a dataset/class based on feasibility flag.
        """
        choosed_feasible_list = self.feasible_prompts[dataset]
        if isinstance(choosed_feasible_list, dict):
            choosed_feasible_list = choosed_feasible_list[_class]

        choosed_infeasible_list = self.infeasible_prompts[dataset]
        if isinstance(choosed_infeasible_list, dict):
            choosed_infeasible_list = choosed_infeasible_list[_class]
        
        sampler = Sampler(self.images_per_real, self.feasibility)

        prompt_words = sampler.sample_words(choosed_feasible_list, choosed_infeasible_list)
        return prompt_words

    def get_dataset_name_for_template(self, dataset):
        """
        Get the dataset label prefix used in prompt generation.
        """
        dataset_name = {
            "imagenet_100": "",
            "imagenet": "",
            "std10": "",
            "oxford_pets": "pet ",
            "fgvc_aircraft": "aircraft ",
            "cars": "car ",
            "eurosat": "satellite ",
            "dtd": "texture ",
            "flowers102": "flower ",
            "food101": "food ",
            "sun397": "scene ",
            "caltech101": "",
            "waterbirds": "bird ",
        }[dataset]
        return dataset_name

    def generate_no_mask(self, prompts, images):
        """
        Generate image without mask guidance using image2image or text2img.
        """
        if self.pipe is None:
            raise RuntimeError('The given args has some errors, you must double check this.')
        images = self.map_np_to_tensor(images)
        if self.with_image:
            images = self.pipe(prompts['prompts'], image=images, strength=0.8, guidance_scale=10.5, num_inference_steps=30).images
        else:
            images = self.pipe(prompt=prompts['prompts'], num_inference_steps=30).images

        final_images = []
        for img in images:
            img = (img * 255).astype(np.uint8)
            final_images.append(img)
        return final_images

    @abstractmethod
    def get_config(self):
        """get the current generation config"""
        pass

    @abstractmethod
    def get_vqa_question(self, preprocess_prompts=None):
        """post process the diffusion output"""
        pass

    def map_np_to_tensor(self, image: List[np.ndarray], mask=False):
        """
        Convert list of np.ndarray (HWC) images or masks to CUDA tensor (BCHW).
        """
        # print('len', len(image))
        # print('image', image[0].shape)
        if not mask:
            if len(image) == 1:
                image = image[0][np.newaxis, ...] 
            else:
                image = np.stack(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = 2.0 * image - 1.0
            image = torch.from_numpy(image)
        else:
            if len(image) == 1:
                image = image[0][np.newaxis, ...] 
            else:
                image = np.stack(image, axis=0)
            image = torch.from_numpy(image)
            image = image.float() / 255.0
        image = image.to("cuda")
        return image


