import random

from .base import BaseGenerator
import torch
import numpy as np
import cv2

from PIL import Image, ImageEnhance, ImageOps
from concurrent.futures import ThreadPoolExecutor
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

class ColorGen(BaseGenerator):

    COLOR_PROMPTS = [
        "a high quality photo of [CLASS] with [COLOR], color change."]

    COLORS = {
        'ruddy': (168, 57, 37),
        'blue gray': (119, 141, 163),
        'silver': (192, 192, 192),
        'red': (255, 0, 0),
        'fawn': (193, 154, 107),
        'white': (255, 255, 255),
        'brindle': (130, 119, 107),
        'brown': (139, 69, 19),
        'black': (0, 0, 0),
        'blue': (0, 0, 255),
        'tan': (210, 180, 140),
        'gold': (255, 215, 0),
        'snow': (255, 250, 250),
        'seal': (36, 33, 28),
        'chocolate': (123, 63, 0),
        'cream': (255, 253, 208),
        'lilac': (200, 162, 200),
        'orange': (255, 165, 0),
        'lemon': (255, 247, 0),
        'liver': (128, 0, 0),
        'roan': (139, 69, 19),
        'champagne': (247, 231, 206),
        'sable': (139, 102, 63),
        'gray': (128, 128, 128),
        'grey': (128, 128, 128),
        'mahogany': (192, 64, 0),
        'sandy': (244, 164, 96),
        'yellow': (255, 255, 0),
        'apricot': (251, 206, 177),
        'biscuit': (233, 194, 166),
        'wheaten': (245, 222, 179),
        'golden': (255, 223, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'neon green': (57, 255, 20),
        'green': (0, 128, 0),
        'lime': (0, 255, 0),
        'turquoise': (64, 224, 208),
        'neon pink': (255, 20, 147),
        'magenta': (255, 0, 255),
        'lavender': (230, 230, 250),
        'chartreuse': (127, 255, 0),
        'hot pink': (255, 105, 180),
        'neon purple': (156, 39, 176),
        'lime green': (50, 205, 50),
        'cyan': (0, 255, 255),
        'neon blue': (77, 77, 255),
        'bright green': (102, 255, 0),
        'bronze': (205, 127, 50),
        'badger': (112, 66, 20),
        'sesame': (247, 199, 128),
        'neon orange': (255, 92, 0),
        'neon yellow': (207, 255, 4),
        'bright yellow': (255, 234, 0),
        'bright orange': (255, 165, 0),
        'bright pink': (255, 20, 147),
        'smoke': (132, 136, 132),
        'teal': (0, 128, 128),
        'beige': (245, 245, 220),
        'peach': (255, 218, 185),    
        'sky blue': (135, 206, 235),  
        'fuchsia': (255, 0, 255),  
        'bright purple': (147, 112, 219), 
        'grey blue': (80, 92, 108),
        'dark beige': (190, 163, 141),    
        'light cyan':(207, 205, 184),
        'light beige': (179, 160, 127),  
        'gull grey':(60, 65, 67),
        'grey cyan':(214,212,197),
        'brown beige':(153, 141, 103),
        'dark pink':(90, 13, 45),
        'light green':(115, 176, 161),
        'olive drab':(107, 142, 35),
        'military green':(117, 130, 77),
        'military gray':(147, 162, 174),
        'dark blue':(27,41,50),
        'deep sea blue': (0, 105, 148),
        'glacier pearl': (234, 237, 237),
        'mars red': (179, 0, 0),
        'super black': (1, 1, 1),
        'jet black': (13, 13, 13),
        'santorini black': (15, 14, 13),
        'snowflake white': (250, 250, 250),
        'cobalt blue': (0, 71, 171),
        'alpine white': (248, 248, 247),
        'denim blue': (21, 96, 189),
        'giallo yellow': (255, 204, 0),
        'brilliant black': (11, 11, 11),
        'ice white': (239, 239, 244),
        'orkney gray': (120, 120, 125),
        'fuji white': (249, 251, 249),
        'true blue': (0, 115, 207),
        'lime green': (50, 205, 50),
        'billet silver': (196, 198, 199),
        'charcoal gray': (54, 69, 79),
        'nautical blue': (0, 70, 127),
        'chrome yellow': (255, 167, 0),
        'obsidian black': (20, 20, 20),
        'anthracite gray': (70, 70, 75),
        'granite crystal': (82, 82, 82),
        'solid black': (2, 2, 2),
        'rally red': (255, 0, 0),
        'crystal white': (245, 245, 245),
        'deep black': (3, 3, 3),
        'gunmetal gray': (76, 76, 76),
        'polar white': (251, 254, 255),
        'mercury silver': (210, 213, 214),
        'shoreline blue': (65, 105, 225),
        'pearl white': (244, 240, 230),
        'cayenne red': (148, 0, 0),
        'slate gray': (112, 128, 144),
        'moonrock silver': (162, 162, 164),
        'platinum silver': (229, 228, 226),
        'deep blue': (0, 0, 139),
        'steel blue': (70, 130, 180),
        'arctic blue': (211, 239, 252),
        'savile gray': (91, 92, 95),
        'brilliant silver': (213, 216, 220),
        'loire blue': (0, 66, 123),
        'bright turquoise': (8, 232, 222),
        'ocean blue': (0, 119, 190),
        'navy blue': (0, 0, 128),
        'flame red': (226, 18, 33),
        'phoenix yellow': (255, 181, 30),
        'predawn gray': (105, 105, 105),
        'crimson red': (220, 20, 60),
        'lunar blue': (70, 120, 180),
        'multi-coat red': (227, 27, 35),
        'indus silver': (210, 210, 210),
        'magnetite gray': (85, 88, 89),
        'bright white': (252, 252, 252),
        'attitude black': (15, 15, 15),
        'moonlight blue': (25, 25, 112),
        'tornado red': (200, 0, 0),
        'lunar gray': (134, 136, 138),
        'electric silver': (215, 215, 215),
        'arctic white': (250, 250, 250),
        'gt silver': (192, 192, 192),
        'united gray': (117, 117, 117),
        'sky blue': (135, 206, 235),
        'reflex silver': (198, 200, 202),
        'candy white': (251, 251, 255),
        'titanium gray': (104, 105, 107),
        'yachting blue': (47, 79, 79),
        'titanium silver': (192, 192, 192),
        'royal blue': (65, 105, 225),
        'matte orange': (255, 165, 0),
        'ivory white': (255, 255, 240),
        'fire red': (179, 0, 0),
        'blizzard pearl': (243, 243, 242),
        'cosmic blue': (0, 78, 146),
        'azure blue': (0, 127, 255),
        'polished silver': (199, 202, 206),
        'silver metallic': (201, 201, 201),
        'deep black pearl': (2, 2, 4),
        'palladium silver': (210, 210, 215),
        'black sapphire': (10, 10, 25),
        'electric purple': (191, 0, 255),
        'steel gray': (119, 136, 153),
        'verde green': (67, 179, 174),
        'midnight black': (4, 4, 4),
        'pepper white': (250, 245, 240),
        'magnetic gray': (79, 85, 87),
        'gun metallic': (78, 81, 83),
        'sunburst yellow': (255, 223, 0),
        'bright blue': (0, 153, 255),
        'super white': (254, 255, 255),
        'electric blue': (125, 249, 255),
        'carmine red': (150, 0, 24),
        'diamond black': (5, 5, 5),
        'jupiter blue': (25, 25, 112),
        'black sand pearl': (25, 25, 25),
        'neon pink': (255, 20, 147),
        'pebble gray': (149, 149, 149),
        'onyx black': (21, 22, 22),
        'glacier white': (246, 247, 247),
        'classic silver': (200, 201, 202),
        'iridium silver': (210, 210, 214),
        'bayside blue': (0, 86, 167),
        'lemon yellow': (255, 250, 102),
        'military olive green': (102, 124, 62),
        'cool silver': (196, 199, 206),
        'championship white': (255, 255, 250),
        'tarmac black': (17, 17, 17),
        'racing green': (0, 66, 37),
        'mystic blue': (16, 78, 139),
        'barcelona red': (176, 0, 32),
        'silver sky': (202, 204, 206),
        'magic blue': (19, 37, 97),
        'rosso red': (189, 16, 26),
        'premium silver': (208, 208, 210),
        'chili red': (208, 17, 23),
        'wicked white': (255, 255, 255),
        'carrara white': (250, 251, 253),
        'volcano red': (217, 33, 33),
        'sapphire blue': (15, 82, 186),
        'diamond white': (252, 255, 250),
        'midnight blue': (25, 25, 112),
        'aspen white': (248, 247, 243),
        # Custom and neon colors
        'bright purple': (153, 50, 204),
        'bright cyan': (0, 255, 255),
        'fluorescent purple': (249, 132, 239),
        'neon green': (57, 255, 20),
        'hot orange': (255, 69, 0),
        'vibrant yellow': (255, 239, 0),
        'fuchsia': (255, 0, 255),
        'vibrant teal': (0, 128, 128),
        'fluorescent orange': (255, 128, 0),
        'hot pink': (255, 105, 180),
        'hot red': (255, 0, 0),
        'vivid green': (0, 255, 0),
        'neon orange': (255, 165, 0),
        'electric green': (0, 255, 0),
        'hot purple': (204, 51, 255),
        'electric turquoise': (0, 244, 255),
        'electric lime': (204, 255, 0),
        'fluorescent lime': (191, 255, 0),
        'neon purple': (148, 0, 211),
        'neon peach': (255, 153, 102),
        'vivid yellow': (255, 255, 0),
        'electric cyan': (0, 255, 255),
        'vivid red': (255, 0, 0),
        'neon teal': (0, 128, 128),
        'neon violet': (148, 0, 211),
        'electric magenta': (255, 0, 255),
        'bright orange': (255, 165, 0),
        'vibrant lime': (50, 205, 50),
        'fluorescent green': (193, 255, 193),
        'fluorescent cyan': (0, 255, 255),
        'vibrant orange': (255, 140, 0),
        'neon blue': (77, 77, 255),
        'electric orange': (255, 104, 31),
        'fluorescent teal': (0, 255, 204),
        'vibrant violet': (191, 62, 255),
        'electric lavender': (244, 187, 255),
        'vivid teal': (0, 128, 128),
        'bright peach': (255, 203, 164),
        'bright violet': (138, 43, 226),
        'electric violet': (143, 0, 255),
        'neon yellow': (255, 255, 0),
        'neon magenta': (255, 0, 255),
        'bright lavender': (230, 230, 250),
        'neon lavender': (182, 102, 210),
        'electric teal': (0, 255, 204),
        'fluorescent red': (255, 91, 71),
        'fluorescent yellow': (255, 255, 102),
        'fluorescent turquoise': (0, 245, 255),
        'electric pink': (255, 20, 147),
        'vivid lavender': (220, 208, 255),
        'bright teal': (0, 128, 128),
        'electric red': (255, 51, 51),
        'vivid cyan': (0, 255, 255),
        "midnight blue": (18, 71, 141),
        "deep purple": (33, 36, 80),
        "light brown":(101, 63, 40),
        "dark red":(89, 26, 37),
        "light cyan":(84, 104, 114),
        "deep green":(62, 74, 70),
        "light green":(115,121,77),
        "deep midnight blue":(32, 37, 77),   
    }


    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        # define the models to be needed
        self.pipe_sdxl = None
        self.compel_sdxl = None

        self.pipe_controlnet = None
        self.compel_controlnet = None

        self._load_pipeline_model()
        # self.config = self.get_config()
        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")


    def generate(self, input_prompts, input_images, mask_images, other_process_dict):
        """
        Perform two-stage image generation:
        1. Use SDXL to generate a base image with color-modified object.
        2. Use ControlNet to refine the appearance based on a canny edge input and IP-Adapter image.

        Args:
            input_prompts (dict): Dict with keys 'prompts', 'negative_prompts', 'conditioning', 'pooled'.
            input_images (torch.Tensor): Input image tensor (BCHW), masked with target object.
            mask_images (torch.Tensor): Corresponding binary masks for the input images.
            other_process_dict (dict): Dictionary containing 'canny_imgs' for ControlNet input.

        Returns:
            List[np.ndarray]: List of generated images.
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
        raw_imgs = raw_imgs[random_index] * 255  # due to two stage generation, we must random choose one here, so for color/texture, we recommand to use bs=1

        ip_white_background = np.ones((1024, 1024, 3), dtype=np.uint8) * 100
        mask_boolean = self.mask == 255
        ip_white_background[mask_boolean, :] = raw_imgs[mask_boolean, :]

        # ip_white_back_pil = Image.fromarray(ip_white_background)
        # ip_white_back_pil.save(f'intermediate_color/{self.tmp_prompt}_composed_prior.png')

        images = self.pipe_controlnet(
            prompt=input_prompts.get('prompts'),
            negative_prompt=input_prompts.get('negative_prompts'),
            prompt_embeds=input_prompts.get('conditioning'),
            pooled_prompt_embeds=input_prompts.get('pooled'),
            image=other_process_dict['canny_imgs'],
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=generator,
            ip_adapter_image=ip_white_background,
            output_type="np",
        ).images
        return images

    def _load_pipeline_model(self):
        if self.use_compel and self.use_mask:
            self.pipe_sdxl, self.compel_sdxl = self.load_sdxl_model()
            self.pipe_controlnet, self.compel_controlnet = self.load_controlnet_model()
        elif not self.use_mask:
            self.pipe = self.load_sdxl_model()
        else:
            self.pipe_sdxl = self.load_sdxl_model()
            self.pipe_controlnet = self.load_controlnet_model()

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
        # low_threshold = 50
        high_threshold = 255
        canny_process_img = image.copy()[:, :, ::-1]

        image = cv2.Canny(canny_process_img, low_threshold, high_threshold)
        image = image[:, :, None]
        canny_image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(canny_image)
        return canny_image
    
    def depth_process(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        # image.save('depth_img.png')
        return image


    def apply_color_advanced(self, image: np.ndarray, mask: np.ndarray, color_name :str, alpha: int=0.6):
        """
        Apply a soft RGB blend of a specified color over the masked region of the image.

        Args:
            image (np.ndarray): Original RGB image.
            mask (np.ndarray): Binary mask where 255 indicates the region to apply color.
            color_name (str): Name of the target color (must be in self.COLORS).
            alpha (float): Blending factor between original and target color.

        Returns:
            Tuple[np.ndarray, PIL.Image, PIL.Image, PIL.Image]:
                - Color-applied image (np.ndarray)
                - Canny edge image (PIL.Image)
                - Solid RGB image (PIL.Image)
                - Preview of color blend (PIL.Image)
        """
        import regex as re

        match = re.search(r'\(([^()]+|(?R))*\)', color_name)
        if match:
            filterd_color_name = match.group(0)[1:-1]  # 去掉外层括号
            # print('matched color', filterd_color_name)

        color = np.array(self.COLORS[filterd_color_name])
        
        rgb_image = Image.new("RGB", (1024,1024), tuple(color))

        h, w, _ = image.shape #TODO: 检查一下维度是否确实这样

        # 保留掩码为255的像素，其他像素设为透明
        mask_boolean = mask == 255
        process_img = image.copy()

        process_img[mask_boolean, :] = (self.image[mask_boolean, :].astype(np.float64) * (1 - alpha) + color.astype(
            np.float64) * alpha).astype(np.uint8)
        
        white_back = np.ones((h, w, 3), dtype=np.uint8) * 255
        mask_boolean = mask == 255
        white_back[mask_boolean, :] = image[mask_boolean, :]
        canny_img = self.canny_process(white_back)

        process_img_pil = Image.fromarray(process_img.astype(np.uint8))
        return process_img, canny_img, rgb_image, process_img_pil

    def pre_process(self, preprocess_prompts=None):
        """
        Prepares inputs for generation by applying the target color over the masked region,
        and extracting corresponding canny edges for ControlNet input.

        Args:
            preprocess_prompts (dict): Must contain 'color' key specifying target color name.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]:
                - Input image tensor
                - Mask tensor
                - Dict with auxiliary inputs including canny edges and RGB reference
        """
        self.config = self.get_config()
        input_image, canny_img, rgb_image, process_img_pil = self.apply_color_advanced(self.image, mask=self.mask, color_name=preprocess_prompts.get('color'),
                                                             alpha=self.config["alpha"])

        input_images = [input_image] * self.batch_size
        mask_images = [self.mask] * self.batch_size

        input_images = self.map_np_to_tensor(input_images)
        mask_images = self.map_np_to_tensor(mask_images, mask=True)
        
        canny_imgs = [canny_img] * self.batch_size # canny images are PILs
        other_process_dict = {'canny_imgs': canny_imgs, 'rgb_image': rgb_image, 'pasted_img': process_img_pil}

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

                post_process_img = post_process_img.astype(np.uint8)
                # post_process_img_pil = Image.fromarray(post_process_img.astype(np.uint8)) #TODO: check if we need to change to rgb
                # final_image.append(post_process_img_pil)
                final_image.append(post_process_img)
        return final_image
    def get_config(self):
        if self.feasibility:
            config_color_feasible = {
                "oxford_pets": {
                    "guidance_scale": 12.0,
                    "num_inference_steps": 20,
                    "strength": 0.3,
                    "alpha": 0.30, },
                "fgvc_aircraft": {
                    "guidance_scale": 12.0,
                    "num_inference_steps": 20,
                    "strength": 0.80,
                    "alpha": 0.6, },
                "cars": {
                    "guidance_scale": 30.0,
                    "num_inference_steps": 20,
                    "strength": 0.85,
                    "alpha": 0.6,
                },
            }
            return config_color_feasible[self.dataset]

        else:
            config_color_infeasible = {
                "oxford_pets": {
                    "guidance_scale": 12.0,
                    "num_inference_steps": 20,
                    "strength": 0.30,
                    "alpha": 0.5, },
                "fgvc_aircraft": {
                    "guidance_scale": 12.0,
                    "num_inference_steps": 20,
                    "strength": 0.80,
                    "alpha": 0.65, },
                "cars": {
                    "guidance_scale": 30.0,
                    "num_inference_steps": 20,
                    "strength": 0.85,
                    "alpha": 0.6,
                },
            }

            return config_color_infeasible[self.dataset]

    def get_prompt(self, gen_index, cur_class):
        """
        Generate a text prompt for the current sample using a color-conditioned template.

        Args:
            gen_index (int): Index into the list of sampled colors.
            cur_class (str): Current object class to insert into the prompt.

        Returns:
            dict: Dictionary with prompt embedding or raw prompt depending on Compel usage,
                as well as the associated color name for later VQA.
        """
        # if self.feasible_prompts is not None and self.infeasible_prompts is not None and gen_index == 0:
        #     self.prompt_words = self.get_base_prompt(self.dataset, cur_class)

        single_prompt_dummy = self.COLOR_PROMPTS[0]  # TODO: could be multiple prompts
        color_this_round = self.prompt_words[gen_index]
        
        current_class_full = str(self.get_dataset_name_for_template(self.dataset) + cur_class)

        single_prompt = single_prompt_dummy.replace('[CLASS]',
                                                    current_class_full)
        single_prompt = single_prompt.replace('[COLOR]', color_this_round)
        
        self.tmp_prompt = single_prompt
        
        self.logger.log(f'Current prompt is {single_prompt}')

        negative_prompt = "bad anatomy, deformed, ugly, disfigured, wrong proportion, low res, low quality"

        prompts = [single_prompt] * self.batch_size
        negative_prompts = [negative_prompt] * self.batch_size

        if self.use_compel and self.use_mask:
            conditioning, pooled = self.compel_controlnet(prompts)
            return {'conditioning': conditioning,
                    'pooled': pooled,
                    'color': color_this_round,
                    'vqa_info': [color_this_round, current_class_full]}

        else:
            return {'prompts': prompts,
                    'negative_prompts': negative_prompts,
                    'color': color_this_round,
                    'vqa_info': [color_this_round, current_class_full]}
            
    def get_vqa_question(self, preprocess_prompts=None):
        current_color = preprocess_prompts.get('vqa_info')[0]
        current_class = preprocess_prompts.get('vqa_info')[1]
        vqa_answer_pairs = [
            {'question': f'Is the {current_class} in the image {current_color}?',
            'choices':['yes', 'no'],
            'answer': 'yes',},
            {'question': f'Does the image show a {current_color} {current_class}?',
            'choices':['yes', 'no'],
            'answer': 'yes',},
            {'question': f'Is the {current_color} possible for the {current_class} to have?',
            'choices':['yes', 'no'],
            'answer': 'yes' if self.feasibility else 'no',},
            {'question': f'Is the {current_color} {current_class} feasibile to appear in the real world?',
            'choices':['yes', 'no'],
            'answer': 'yes' if self.feasibility else 'no',},         
                            ]
        return vqa_answer_pairs
        

    def get_prompt_batched(self, cur_class):
        """get the current generation prompt"""
        self.batch_size = self.images_per_real # kind of hard code here
                
        # if self.feasible_prompts is not None and self.infeasible_prompts is not None:
        #     self.prompt_words = self.get_base_prompt(self.dataset, cur_class)
        
        batched_prompts = []

        single_prompt_dummy = self.COLOR_PROMPTS[0]  # TODO: could be multiple prompts
        
        current_class_full = str(self.get_dataset_name_for_template(self.dataset) + cur_class)
        
        for index in range(self.images_per_real):
            color_this_round = self.prompt_words[index]
            single_prompt = single_prompt_dummy.replace('[CLASS]',
                                                        current_class_full)
            single_prompt = single_prompt.replace('[COLOR]', color_this_round)
            batched_prompts.append(single_prompt)
            
        # TODO：都改成一样的！！！
        negative_prompt = "bad anatomy, deformed, ugly, disfigured, wrong proportion, low res, low quality"

        prompts = batched_prompts
        negative_prompts = [negative_prompt] * self.images_per_real

        if self.use_compel and self.use_mask:
            # conditioning = self.compel_controlnet(prompts)

            conditioning, pooled = self.compel_sdxl(prompts)
            return {'conditioning': conditioning,
                    'pooled': pooled,
                    'color': self.prompt_words,
                    'vqa_info': [self.prompt_words, current_class_full]}

        else:
            return {'prompts': prompts,
                    'negative_prompts': negative_prompts,
                    'color': self.prompt_words,
                    'vqa_info': [self.prompt_words, current_class_full]}
            
        
    def pre_process_batched(self, preprocess_prompts=None):
        
        self.config = self.get_config()
        color_names = preprocess_prompts.get('color')
        
        input_images = []
        
        for color_name in color_names:
            input_image, canny_img, rgb_image, process_img_pil = self.apply_color_advanced(self.image, mask=self.mask, color_name=color_name,
                                                             alpha=self.config["alpha"])
            
            input_images.append(input_image)
        
        canny_imgs = [canny_img]
            
        mask_images = [self.mask] * self.batch_size

        input_images = self.map_np_to_tensor(input_images)
        mask_images = self.map_np_to_tensor(mask_images, mask=True)
        other_process_dict = {'canny_imgs': canny_imgs}

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
        raw_imgs = raw_imgs * 255  # due to two stage generation, we must random choose one here, so for color/texture, we recommand to use bs=1

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
            images_list.append(images.squeeze())     

        return images_list

        
