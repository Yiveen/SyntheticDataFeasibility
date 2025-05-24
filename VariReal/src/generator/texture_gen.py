import random
from .base import BaseGenerator
import torch
import numpy as np
import cv2
from PIL import Image
import regex as re 

from diffusers import DiffusionPipeline, AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoPipelineForText2Image, DPMSolverMultistepScheduler

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
)

from compel import Compel, ReturnedEmbeddingsType

class TextureGen(BaseGenerator):

    TEXTURE_PROMPTS = [
        "a high quality photo of [CLASS] with detailed [TEXTURE], surface texture change."]
    
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


    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        # define the models to be needed
        self.pipe_sdxl = None
        self.compel_sdxl = None

        self.pipe_sd2 = None
        self.compel_sd2 = None

        self.pipe_controlnet = None
        self.compel_controlnet = None
        
        self.use_full = False
        
        self._load_pipeline_model()
        
    def load_controlnet_model(self):
        self.logger.log("### Loading the SDXL Controlnet ###")
        controlnets = [
            ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16,
                cache_dir=self.cache_dir
            ),
        ]

        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
                                            cache_dir=self.cache_dir)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", controlnet=controlnets, vae=vae,
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
        if self.feasibility:
            if self.dataset == 'oxford_pets':
                pipe.set_ip_adapter_scale(0.2)
            elif self.dataset == 'fgvc_aircraft':
                pipe.set_ip_adapter_scale(0.65)
            elif self.dataset == 'cars':
                pipe.set_ip_adapter_scale(0.65)
            else:
                pipe.set_ip_adapter_scale(0.4)
        else:
            if self.dataset == 'oxford_pets':
                pipe.set_ip_adapter_scale(0.5)
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

            return pipe, compel
        else:
            return pipe

    def generate(self, input_prompts, input_images, mask_images, other_process_dict):
        """
        Perform two-stage generation:
        1. Use SDXL inpainting to generate a base image with new surface texture.
        2. Use ControlNet + IP-Adapter to refine the result using edge guidance and masked blending.

        Args:
            input_prompts (dict): Includes 'prompts', 'negative_prompts', 'conditioning', 'pooled'.
            input_images (torch.Tensor): Image tensor with masked regions for inpainting.
            mask_images (torch.Tensor): Binary mask tensor for inpainting region.
            other_process_dict (dict): Includes canny edge maps and intermediate image info.

        Returns:
            List[np.ndarray]: List of generated RGB images.
        """
        generator = self._get_generator()
        raw_imgs = self.pipe_sdxl(
            prompt=input_prompts.get('prompts'),
            negative_prompt=input_prompts.get('negative_prompts'),
            prompt_embeds=input_prompts.get('conditioning'),
            pooled_prompt_embeds=input_prompts.get('pooled'),
            image=input_images,
            mask_image=mask_images,
            guidance_scale=self.config['guidance_scale'],
            # affects the alignment between the text prompt and the generated image.
            num_inference_steps=self.config['num_inference_steps'],  # steps between 15 and 30 work well for us
            strength=self.config['strength'],  # make sure to use `strength` below 1.0
            generator=generator,
            padding_mask_crop=None,  # Crops masked area + padding from image and mask.
            output_type="np",
        ).images

        
        random_index = np.random.randint(0, self.batch_size)
        raw_imgs = raw_imgs[
            random_index] * 255 # due to two stage generation, we must random choose one here, so for color/texture, we recommand to use bs=1
        
        ip_white_background = np.ones((1024, 1024, 3), dtype=np.uint8) * 100
        # print('mask', self.mask)
        mask_boolean = self.mask == 255
        ip_white_background[mask_boolean, :] = raw_imgs[mask_boolean, :]

        images = self.pipe_controlnet(
            prompt=input_prompts.get('prompts'),
            negative_prompt=input_prompts.get('negative_prompts'),
            prompt_embeds=input_prompts.get('conditioning'),
            pooled_prompt_embeds=input_prompts.get('pooled'),
            image=other_process_dict['canny_imgs'],
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=generator,
            ip_adapter_image=[ip_white_background],
            output_type="np",
        ).images
        

        return images


    def _load_pipeline_model(self):
            if self.use_compel and self.use_mask:
                self.pipe_sdxl, self.compel_sdxl = self.load_sdxl_model()
                self.pipe_sd2, self.compel_sd2 = self.load_stable_diffision2()
                self.pipe_controlnet, self.compel_controlnet = self.load_controlnet_model()
            elif not self.use_mask:
                self.pipe = self.load_sdxl_model()
            else:
                self.pipe_sdxl = self.load_sdxl_model()
                self.pipe_controlnet = self.load_controlnet_model()
                self.pipe_sd2 = self.load_stable_diffision2()
                
    def canny_process(self, image):
        """
        Applies Canny edge detection on a BGR image and returns an RGB PIL image.

        Args:
            image: Input BGR image.

        Returns:
            PIL.Image: RGB image with Canny edges.
        """
        blurred_img = cv2.blur(image,ksize=(5,5))
        med_val = np.median(blurred_img) 
        low_threshold = int(max(0 ,0.7*med_val))
        high_threshold = 255
        canny_process_img = image.copy()[:, :, ::-1]

        image = cv2.Canny(canny_process_img, low_threshold, high_threshold)
        image = image[:, :, None]
        canny_image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(np.array(canny_image).astype('uint8'))
        return canny_image

    def apply_texture_advanced(self, image: np.ndarray, mask: np.ndarray, texture_img:np.ndarray, alpha: int=0.6):
        """
        Apply texture blending over a masked object region with optional sharpening.

        Args:
            image (np.ndarray): Original RGB image.
            mask (np.ndarray): Binary object mask.
            texture_img (np.ndarray): Texture image to blend in.
            alpha (float): Blending factor between original and texture.

        Returns:
            Tuple[np.ndarray, PIL.Image]:
                - Blended image with texture.
                - Canny edge image for ControlNet input.
        """
        h, w, _ = image.shape 

        mask_boolean = mask == 255
        process_img = image.copy()
        
        process_img[mask_boolean, :] = (image[mask_boolean, :].astype(np.float64) * (1 - alpha) + texture_img[mask_boolean, :].astype(np.float64) * alpha).astype(np.uint8)

        white_back = np.ones((h, w, 3), dtype=np.uint8) * 255
        white_back[mask_boolean, :] = image[mask_boolean, :]
        
        # process_img_pil = Image.fromarray(process_img.astype(np.uint8))
        # process_img_pil.save('process_img.png')

        canny_img = self.canny_process(white_back)
        
        return process_img, canny_img

    def pre_process(self, preprocess_prompts=None):
        self.config = self.get_config()
        texture_img = self.pipe_sd2(prompt=preprocess_prompts.get("preprocess_prompts").get("sd2_prompt"), negative_prompt= ["clothes, objects, figures, shapes, symbols, letters, numbers, text, landscapes, scenes, animals, humans, artifacts, parts"] * self.batch_size,
                                    prompt_embeds=preprocess_prompts.get("preprocess_prompts").get("sd2_embeds"), num_inference_steps=15, output_type="np", height=1024,  
                                                    width=1024).images
        texture_img = texture_img.squeeze()  
        texture_img = (texture_img * 255).round().astype("uint8")  
        
        h, w, _ = texture_img.shape
        top_left_x = np.random.randint(0, w - 256 + 1)
        top_left_y = np.random.randint(0, h - 256 + 1)
        selected_region = texture_img[top_left_y:top_left_y + 256, top_left_x:top_left_x + 256]

        enlarged_region = cv2.resize(selected_region, (1024, 1024), interpolation=cv2.INTER_CUBIC)

        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        cropped_texture_img = cv2.filter2D(enlarged_region, -1, kernel)
        
        texture_img_pil, input_image_pil = None, None

        input_image, canny_img = self.apply_texture_advanced(self.image, mask=self.mask, texture_img=cropped_texture_img,
                                                            alpha=self.config["alpha"])
        input_images = [input_image] * self.batch_size
        mask_images = [self.mask] * self.batch_size

        input_images = self.map_np_to_tensor(input_images)
        mask_images = self.map_np_to_tensor(mask_images, mask=True)

        canny_imgs = [canny_img] # canny images are PILs
        other_process_dict = {'canny_imgs': canny_imgs, 'sd2_texture': texture_img_pil, 'pasted_img': input_image_pil}

        return input_images, mask_images, other_process_dict

    def post_process(self, images):
        final_image = []
        zero_one_mask = self.mask / 255.0
        zero_one_mask_3d = np.expand_dims(zero_one_mask, axis=-1)
        zero_one_mask_3d = np.repeat(zero_one_mask_3d, 3, axis=-1)
        if self.use_mask:
            for image in images:
                image = image * 255
                post_process_img = image.astype(np.float32) * (zero_one_mask_3d) + self.image.astype(
                        np.float32) * (1 - zero_one_mask_3d)
                final_image.append(post_process_img)
        return final_image
    
    def get_config(self):
        if self.feasibility:
            config_texture_feasible = {
                "oxford_pets": {
                    "guidance_scale": 12.0,
                    "num_inference_steps": 20,
                    "strength": 0.3,
                    "alpha": 0.5, },
                "fgvc_aircraft": {
                    "guidance_scale": 8.0,
                    "num_inference_steps": 20,
                    "strength": 0.65,
                    "alpha": 0.5, },
                "cars": {
                    "guidance_scale": 30.0,
                    "num_inference_steps": 20,
                    "strength": 0.65,
                    "alpha": 0.65,
                },
            }
            return config_texture_feasible[self.dataset]

        else:            
            config_texture_infeasible = {
                "oxford_pets": {
                    "guidance_scale": 12.0,
                    "num_inference_steps": 20,
                    "strength": 0.3,
                    "alpha": 0.40, },
                "fgvc_aircraft": {
                    "guidance_scale": 8.0,
                    "num_inference_steps": 20,
                    "strength": 0.3,
                    "alpha": 0.65, },
                "cars": {
                    "guidance_scale": 8.0,
                    "num_inference_steps": 20,
                    "strength": 0.3,
                    "alpha": 0.65,
                },
            }

            return config_texture_infeasible[self.dataset]

    def get_prompt(self, gen_index, cur_class):
        """
        Construct both:
        - SDXL prompt for object-texture synthesis.
        - SD2 prompt for background texture synthesis.

        Args:
            gen_index (int): Index into the sampled texture list.
            cur_class (str): Object class name.

        Returns:
            dict: Including prompts, embeds, and metadata for VQA evaluation.
        """
        single_prompt_dummy = self.TEXTURE_PROMPTS[0]  # TODO: could be multiple prompts
        texture_this_round = self.prompt_words[gen_index]
        
        # texture_this_round = re.search(r'\((.*?)\)', texture_this_round).group(1)
        sd2_texture_this_round = "a solid color texture of " + texture_this_round.split(':')[0] + ", abstract, no objects, no figures, no shapes, minimalistic pattern, consistent texture, continuous surface"

        
        current_class_full = str(self.get_dataset_name_for_template(self.dataset) + cur_class)

        single_prompt = single_prompt_dummy.replace('[CLASS]',
                                                    current_class_full)
        single_prompt = single_prompt.replace('[TEXTURE]', texture_this_round)
        
        self.tmp_prompt = texture_this_round.split(':')[0]
        
        self.logger.log(f'Current prompt is {single_prompt}')

        # TODO：都改成一样的！！！
        negative_prompt = "bad anatomy, deformed, ugly, disfigured, wrong proportion, low res, low quality"

        prompts = [single_prompt] * self.batch_size
        negative_prompts = [negative_prompt] * self.batch_size
        sd2_back_this_round_prompts = [sd2_texture_this_round] * self.batch_size

        if self.use_compel and self.use_mask:
            conditioning, pooled = self.compel_sdxl(prompts)
            prompt_embeds_sd2 = self.compel_sd2(sd2_back_this_round_prompts)
            return {'conditioning': conditioning,
                    'pooled': pooled,
                    'preprocess_prompts': {'name': texture_this_round,
                                        'sd2_embeds': prompt_embeds_sd2},
                    'vqa_info': [texture_this_round, current_class_full]}

        else:
            return {'prompts': prompts,
                    'negative_prompts': negative_prompts,
                    'preprocess_prompts': {'name':texture_this_round,
                                        'sd2_prompts': sd2_back_this_round_prompts},
                    'vqa_info': [texture_this_round, current_class_full]}
    
    def get_vqa_question(self, preprocess_prompts=None):
        current_texture = preprocess_prompts.get('vqa_info')[0]
        current_class = preprocess_prompts.get('vqa_info')[1]
        vqa_answer_pairs = [
            {'question': f'Is the {current_class} in the image {current_texture}?',
            'choices':['yes', 'no'],
            'answer': 'yes',},
            {'question': f'Does the image show a {current_texture} {current_class}?',
            'choices':['yes', 'no'],
            'answer': 'yes',},
            {'question': f'Is the {current_texture} possible for the {current_class} to have?',
            'choices':['yes', 'no'],
            'answer': 'yes' if self.feasibility else 'no',},
            {'question': f'Is the {current_texture} {current_class} feasibile to appear in the real world?',
            'choices':['yes', 'no'],
            'answer': 'yes' if self.feasibility else 'no',},         
                            ]
        return vqa_answer_pairs
    
    
    def get_prompt_batched(self, cur_class):
        """get the current generation prompt"""
        self.batch_size = self.images_per_real # kind of hard code here
        # if self.feasible_prompts is not None and self.infeasible_prompts is not None:
        #     if self.dataset == 'fgvc_aircraft':
        #         cur_class_catgory = self.CATEGORY_MAP[cur_class]
        #     else:
        #         cur_class_catgory = cur_class
        #     self.prompt_words = self.get_base_prompt(self.dataset, cur_class_catgory)

        single_prompt_dummy = self.TEXTURE_PROMPTS[0]  # TODO: could be multiple prompts
        current_class_full = str(self.get_dataset_name_for_template(self.dataset) + cur_class)
        
        prompts = []
        sd2_back_this_round_prompts = []
        
        for index in range(self.images_per_real):
            texture_this_round = self.prompt_words[index]
            
            # texture_this_round = re.search(r'\((.*?)\)', texture_this_round).group(1)
            sd2_texture_this_round = "a solid color texture of " + texture_this_round.split(":")[0] + ", abstract, no objects, no figures, no shapes, minimalistic pattern, consistent texture, continuous surface"

            single_prompt = single_prompt_dummy.replace('[CLASS]',
                                                        current_class_full)
            single_prompt = single_prompt.replace('[TEXTURE]', texture_this_round)
            sd2_back_this_round_prompts.append(sd2_texture_this_round)
            prompts.append(single_prompt)
            
        negative_prompt = "bad anatomy, deformed, ugly, disfigured, wrong proportion, low res, low quality"

        negative_prompts = [negative_prompt] * self.images_per_real
        
        # print(len(prompts), len(negative_prompts), len(sd2_back_this_round_prompts))

        if self.use_compel and self.use_mask:
            conditioning, pooled = self.compel_sdxl(prompts)
            prompt_embeds_sd2 = self.compel_sd2(sd2_back_this_round_prompts)
            return {'conditioning': conditioning,
                    'pooled': pooled,
                    'preprocess_prompts': {'name': self.prompt_words,
                                        'sd2_embeds': prompt_embeds_sd2},
                    'vqa_info': [self.prompt_words, current_class_full]}

        else:
            return {'prompts': prompts,
                    'negative_prompts': negative_prompts,
                    'preprocess_prompts': {'name':self.prompt_words,
                                        'sd2_prompts': sd2_back_this_round_prompts},
                    'vqa_info': [self.prompt_words, current_class_full]}
            
        
    def pre_process_batched(self, preprocess_prompts=None):
        
        self.config = self.get_config()
        texture_imgs = self.pipe_sd2(prompt=preprocess_prompts.get("preprocess_prompts").get("sd2_prompt"), negative_prompt= ["clothes, objects, figures, shapes, symbols, letters, numbers, text, landscapes, scenes, animals, humans, artifacts"] * self.batch_size,
                                    prompt_embeds=preprocess_prompts.get("preprocess_prompts").get("sd2_embeds"), num_inference_steps=12, output_type="np", height=1024,  
                                                    width=1024).images

        texture_imgs = (texture_imgs * 255).round().astype("uint8")
        
        cropped_texture_imgs = []

        for texture_img in texture_imgs:
            h, w, _ = texture_img.shape
            top_left_x = np.random.randint(0, w - 256 + 1)
            top_left_y = np.random.randint(0, h - 256 + 1)
            
            selected_region = texture_img[top_left_y:top_left_y + 256, top_left_x:top_left_x + 256]

            enlarged_region = cv2.resize(selected_region, (1024, 1024), interpolation=cv2.INTER_CUBIC)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened_region = cv2.filter2D(enlarged_region, -1, kernel)
            cropped_texture_imgs.append(sharpened_region)
        
        input_images = []
        for texture_img in cropped_texture_imgs:
        
            input_image, canny_img = self.apply_texture_advanced(self.image, mask=self.mask, texture_img=texture_img,
                                                            alpha=self.config["alpha"])
            input_images.append(input_image)
        canny_img = [canny_img]

        
        mask_images = [self.mask] * self.batch_size

        input_images = self.map_np_to_tensor(input_images)
        mask_images = self.map_np_to_tensor(mask_images, mask=True)

        other_process_dict = {'canny_imgs': canny_img}

        return input_images, mask_images, other_process_dict
    
    def generate_batched(self, input_prompts, input_images, mask_images, other_process_dict):
        """method to use pipeline to generate images"""
        generator = self._get_generator()
        raw_imgs = self.pipe_sdxl(
            prompt=input_prompts.get('prompts'),
            negative_prompt=input_prompts.get('negative_prompts'),
            prompt_embeds=input_prompts.get('conditioning'),
            pooled_prompt_embeds=input_prompts.get('pooled'),
            image=input_images,
            mask_image=mask_images,
            guidance_scale=self.config['guidance_scale'],
            # affects the alignment between the text prompt and the generated image.
            num_inference_steps=self.config['num_inference_steps'],  # steps between 15 and 30 work well for us
            strength=self.config['strength'],  # make sure to use `strength` below 1.0
            generator=generator,
            padding_mask_crop=None,  # Crops masked area + padding from image and mask.
            output_type="np",
        ).images

        raw_imgs = raw_imgs * 255 # due to two stage generation, we must random choose one here, so for color/texture, we recommand to use bs=1
        ip_white_background = np.ones((self.batch_size, 1024, 1024, 3), dtype=np.uint8) * 100
        mask_boolean = self.mask == 255
        ip_white_background[:, mask_boolean, :] = raw_imgs[:, mask_boolean, :]

        
        images_list = []
        # print(ip_white_background[0].shape)
        
        for gen_i in range(self.images_per_real):
            images = self.pipe_controlnet(
                prompt=input_prompts.get('prompts'),
                negative_prompt=input_prompts.get('negative_prompts'),
                prompt_embeds=input_prompts.get('conditioning')[gen_i].unsqueeze(0),
                pooled_prompt_embeds=input_prompts.get('pooled')[gen_i].unsqueeze(0),
                image=other_process_dict['canny_imgs'],
                guidance_scale=7.5,
                num_inference_steps=20,
                generator=generator,
                ip_adapter_image=[ip_white_background[gen_i]],
                output_type="np",
            ).images
            # print('gen_images', images.squeeze().shape)
            images_list.append(images.squeeze())

        return images_list
