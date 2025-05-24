from .base import BaseGenerator
import torch
import numpy as np
import cv2
import random
from PIL import Image

class BackGen(BaseGenerator):

    """
    Background editing generator using SDXL and Stable Diffusion 2 pipelines.
    Generates feasible/infeasible background-conditioned images for a given object class.
    """

    BACKGROUND_PROMPTS = [
        "a high quality photo that [CLASS] is harmonious on the [BACKGROUND]"]

    def __init__(self, *args, **kwds):
        """
        Initializes pipelines and compels, then loads model based on configuration.
        """
        super().__init__(*args, **kwds)
        # define the models to be needed 
        self.pipe_sdxl = None
        self.compel_sdxl = None
        self.pipe_sd2 = None
        self.compel_sd2 = None

        self._load_pipeline_model()
        # self.prompt_dict = self.get_prompt()


    def generate(self, input_prompts, input_images, mask_images, other_process_dict):
        """
        Runs SDXL inpainting pipeline on masked input image.

        Returns:
            np.ndarray: Generated images.
        """
        generator = self._get_generator()
        images = self.pipe_sdxl(
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

        return images

    def _load_pipeline_model(self):
        """
        Loads required pipelines depending on mode (masking, compel, etc.)
        """
        if self.use_compel and self.use_mask or self.give_post:
            self.pipe_sdxl, self.compel_sdxl = self.load_sdxl_model()
            self.pipe_sd2, self.compel_sd2 = self.load_stable_diffision2()
        elif not self.use_mask:
            self.pipe = self.load_sdxl_model()
        else:
            self.pipe_sdxl = self.load_sdxl_model()
            self.pipe_sd2 = self.load_stable_diffision2()

    def apply_mask_advanced(self, image: np.ndarray, mask: np.ndarray, initial_back: np.ndarray, dilated_factor: int=60):
        """
        Overlays the object on a generated background by dilating the mask.

        Args:
            image (np.ndarray): Original image with object.
            mask (np.ndarray): Binary mask of object region.
            initial_back (np.ndarray): Generated background.
            dilated_factor (int): Size of dilation kernel.

        Returns:
            np.ndarray: Composite image with replaced background.
        """
        
        # kernel size
        kernel_size = dilated_factor  
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # dilated
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        dilated_mask_inverted = 255 - dilated_mask
        mask_boolean = (dilated_mask_inverted == 255)
        process_img = image.copy()
        process_img[mask_boolean, :] = initial_back[mask_boolean, :]
        
        # process_img_pil = Image.fromarray(process_img.astype(np.uint8))
        # process_img_pil.save(f'intermediate_back_dilate0/{self.tmp_prompt}_process_img_back.png')
        
        return process_img
    
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

    def pre_process(self, preprocess_prompts=None):
        """
        Generates a background image with SD2 and pastes the original object using mask blending.

        Returns:
            (torch.Tensor, torch.Tensor, Dict): Processed image tensor, mask tensor, debug info.
        """
        self.config = self.get_config() # for each generation time, we have different configs
        if preprocess_prompts is None:
            raise  RuntimeError('You must give stable diffusion prompts/embeddings firstly.')

        height, width, _ = self.image.shape
        # generate the backgrounds
        image_initial_backgrounds = self.pipe_sd2(prompt=preprocess_prompts.get('preprocess_prompts'), prompt_embeds=preprocess_prompts.get('preprocess_embeds'), num_inference_steps=20, output_type="np", height=1024,  
                                                    width=1024).images
        image_initial_backgrounds = (image_initial_backgrounds * 255).round().astype("uint8")
        
        image_initial_backgrounds_pil = Image.fromarray(image_initial_backgrounds[0])
        # image_initial_backgrounds_pil.save(f'intermediate_back_dilate0/{self.tmp_prompt}_image_initial_backgrounds_pil.png')
        # Assume `image_arrays` is a list of images and `masks` is a list of masks for the batch
        input_images = [
            self.apply_mask_advanced(self.image, mask=self.mask, initial_back=image_initial_background,
                                dilated_factor=self.config["dilated_factor"])
            for image_initial_background in
            image_initial_backgrounds
        ]

        input_images_ts = self.map_np_to_tensor(input_images) # input image is rgb

        inverted_mask = 255. - self.mask
        mask_blurred = cv2.GaussianBlur(inverted_mask, (41, 41), 0)
        mask_blurred_pil = Image.fromarray(mask_blurred).convert("L")
        # mask_blurred_pil.save(f'intermediate_back_dilate0/{self.tmp_prompt}_mask_blurred.png')
        
        mask_images = [mask_blurred] * self.batch_size
        mask_images = self.map_np_to_tensor(mask_images, mask=True)
  
        other_process_dict = {'pasted_img': Image.fromarray(input_images[0]),'sd_2_background': image_initial_backgrounds_pil}
        return input_images_ts, mask_images, other_process_dict

    def post_process(self, images):
        """
        Composites the original object back into the generated image (if masking was used).

        Returns:
            List[np.ndarray]: Final blended images.
        """
        final_image = []
        zero_one_mask = self.mask / 255.0
        zero_one_mask_3d = np.expand_dims(zero_one_mask, axis=-1)
        zero_one_mask_3d = np.repeat(zero_one_mask_3d, 3, axis=-1)
        if self.use_mask:
            for image in images:
                image = image * 255
                post_process_img = image.astype(np.float32) * (1. - zero_one_mask_3d) + self.image.astype(np.float32) * zero_one_mask_3d
                post_process_img = post_process_img.astype(np.uint8)
                # post_process_img_pil = Image.fromarray(post_process_img.astype(np.uint8)) #TODO: check if we need to change to rgb
                # final_image.append(post_process_img_pil)
                final_image.append(post_process_img)
        return final_image

    def get_config(self):
        """
        Returns SDXL generation parameters tuned per dataset.
        """
        config_back = {
            "oxford_pets": {
                "guidance_scale": 40,
                "num_inference_steps": 30,
                "strength": 0.99,
                "dilated_factor": 120,
                "lora_scale": 0.9,
            },
            "fgvc_aircraft": {
                "guidance_scale": 7.5,
                "num_inference_steps": 30,
                "strength": 0.95,
                "dilated_factor": 50,
                "lora_scale": 0.9,
            },
            "cars": {
                "guidance_scale": 7.5,
                "num_inference_steps": 30,
                "strength": 0.90,
                "dilated_factor": 25,
                "lora_scale": 0.9,
            },
            "waterbirds": {
                "guidance_scale": 7.5,
                "num_inference_steps": 30,
                "strength": 0.80,
                "dilated_factor": 0,
                "lora_scale": 0.9,
            },
        }
        return config_back[self.dataset]

    def get_prompt(self, gen_index, cur_class):
        # if self.feasible_prompts is not None and self.infeasible_prompts is not None:
        #     self.prompt_words = self.get_base_prompt(self.dataset, cur_class)

        single_prompt_dummy = self.BACKGROUND_PROMPTS[0]  # TODO: could be multiple prompts
                    
        back_this_round = self.prompt_words[gen_index]
        
        sd2_back_this_round = f"A photo of pure background without {self.general_class} on" + back_this_round
        
        current_class_full = str(self.get_dataset_name_for_template(self.dataset) + cur_class)

        single_prompt = single_prompt_dummy.replace('[CLASS]', current_class_full)
        if self.use_mask:
            single_prompt = single_prompt.replace('[BACKGROUND]', back_this_round)
        else:
            single_prompt = single_prompt.replace('[BACKGROUND]', back_this_round.split(':')[0])
            
        self.tmp_prompt = back_this_round.split(':')[0]
        import os
        os.makedirs('intermediate_back_dilate0/', exist_ok=True)

        self.logger.log(f'Current prompt is {single_prompt}')

        negative_prompt = "similar, no distinction, bad anatomy, deformed, ugly, disfigured, wrong proportion, low quality"

        prompts = [single_prompt] * self.batch_size
        negative_prompts = [negative_prompt] * self.batch_size
        sd2_back_this_round_prompts = [sd2_back_this_round] * self.batch_size

        if self.use_compel and self.use_mask:
            prompt_embeds_sd2 = self.compel_sd2(sd2_back_this_round_prompts)
            conditioning, pooled = self.compel_sdxl(prompts)
            return {'conditioning': conditioning,
                    'pooled': pooled,
                    'preprocess_embeds': prompt_embeds_sd2,
                    'vqa_info': [back_this_round.split(':')[0], current_class_full]}

        else:
            return {'prompts': prompts,
                    'negative_prompts': negative_prompts,
                    'preprocess_prompts': sd2_back_this_round_prompts,
                    'vqa_info': [back_this_round.split(':')[0], current_class_full]}
            
    def get_vqa_question(self, preprocess_prompts=None):
        """
        Creates VQA-style question-answer pairs for downstream evaluation.
        """
        current_background = preprocess_prompts.get('vqa_info')[0]
        current_class = preprocess_prompts.get('vqa_info')[1]
        vqa_answer_pairs = [
            {'question': f'Is the object of the image in {current_background} background?',
            'choices':['yes', 'no'],
            'answer': 'yes',},
            {'question': f'Does the image background is {current_background}?',
            'choices':['yes', 'no'],
            'answer': 'yes',},
            {'question': f'Dose the {current_background} look feasible for the {current_class}?',
            'choices':['yes', 'no'],
            'answer': 'yes' if self.feasibility else 'no',},
            {'question': f'Is it possible for the {current_class} in this image to appear in the real world with its background?',
            'choices':['yes', 'no'],
            'answer': 'yes' if self.feasibility else 'no',},         
                            ]
        return vqa_answer_pairs
    

    def get_prompt_batched(self, cur_class):
        """
        Constructs prompts for batched generation.

        Returns:
            Dict: Batched prompts and embeddings.
        """
        """get the current generation prompt"""
        self.batch_size = self.images_per_real # kind of hard code here
        # if self.feasible_prompts is not None and self.infeasible_prompts is not None:
        #     self.prompt_words = self.get_base_prompt(self.dataset, cur_class)

        single_prompt_dummy = self.BACKGROUND_PROMPTS[0]  # TODO: could be multiple prompts
        lighntning_prompt = None
                
        batched_prompts = []
        sd2_batched_prompts = []
        for index in range(self.images_per_real):     
            back_this_round = self.prompt_words[index]       
            if lighntning_prompt is not None:
                back_this_round = back_this_round.split(':')[0] + f',{lighntning_prompt[index]}, contrasts, :' + back_this_round.split(':')[1] 
        
            sd2_back_this_round = f"A photo of pure background without {self.general_class} on" + back_this_round
        
            current_class_full = str(self.get_dataset_name_for_template(self.dataset) + cur_class)

            single_prompt = single_prompt_dummy.replace('[CLASS]', current_class_full)
            if self.use_mask:
                single_prompt = single_prompt.replace('[BACKGROUND]', back_this_round)
            else:
                single_prompt = single_prompt.replace('[BACKGROUND]', back_this_round.split(':')[0])
                
            batched_prompts.append(single_prompt)
            sd2_batched_prompts.append(sd2_back_this_round)

        negative_prompt = "similar, no distinction, bad anatomy, deformed, ugly, disfigured, wrong proportion, low quality"

        prompts = batched_prompts
        negative_prompts = [negative_prompt] * self.images_per_real
        sd2_back_this_round_prompts = sd2_batched_prompts

        if self.use_compel and self.use_mask:
            prompt_embeds_sd2 = self.compel_sd2(sd2_back_this_round_prompts)
            conditioning, pooled = self.compel_sdxl(prompts)
            return {'conditioning': conditioning,
                    'pooled': pooled,
                    'preprocess_embeds': prompt_embeds_sd2,
                    'vqa_info': [back_this_round.split(':')[0], current_class_full]}

        else:
            return {'prompts': prompts,
                    'negative_prompts': negative_prompts,
                    'preprocess_prompts': sd2_back_this_round_prompts,
                    'vqa_info': [back_this_round.split(':')[0], current_class_full]}
            
    def pre_process_batched(self, preprocess_prompts=None):
        """pre process the input image and mask"""
        return self.pre_process(preprocess_prompts)

    def generate_batched(self, input_prompts, input_images, mask_images, other_process_dict):
        """method to use pipeline to generate images"""
        return self.generate(input_prompts, input_images, mask_images, other_process_dict)
        

