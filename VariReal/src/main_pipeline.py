import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import pickle
import random
import glob
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as tfms
from tqdm import tqdm
import open_clip
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
import re
import time
import yaml

# from utils.template import FEASIBLE_BACK, INFEASIBLE_BACK, FEASIBLE_COLOR, FEASIBLE_TEXTURE, INFEASIBLE_COLOR, INFEASIBLE_TEXTURE
from config.config_back import FEASIBLE_BACK, INFEASIBLE_BACK
from config.config_color import FEASIBLE_COLOR, INFEASIBLE_COLOR
from config.config_texture import FEASIBLE_TEXTURE, INFEASIBLE_TEXTURE
from Finetune.classify.data import get_data_loader
from detector.grounding_mask import GroundingMask
from generator.back_gen import BackGen
from generator.color_gen import ColorGen
from generator.texture_gen import TextureGen   
from vlm_filter.vqa_models import VQAModelFiltering
from utils.utils import get_generated_key, process_and_save_image, get_dataset_name_for_template, create_out_dir, general_preprocess, general_postprocess, \
    group_outputs_for_batch_repeats
from utils.logger import Logger
from utils.config import GENERAL_CALSS_MAPPING
from Finetune.classify.util_data import SUBSET_NAMES


def main(args, debug):
    REPEAT = 1
    
    logger = Logger(args)
    logger.log("Start the generation process.")
    start_time = time.time()
    
    if torch.cuda.is_available():
        logger.log(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.log("Using CPU")
    
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    
    #### Generation related ####
    dataset = args.dataset
    batch_size = args.batch_size
    images_per_real = args.images_per_real
    n_img_per_cls = args.n_img_per_cls
    use_mask = not args.no_mask
    with_image = args.with_image
    gen_class = args.gen_class
    general_class = GENERAL_CALSS_MAPPING[dataset]
    back = args.back
    color = args.color
    texture = args.texture
    objsize = args.objsize
    feasible = args.feasiblity
    infeasible = not args.feasiblity

    yaml_file = args.yaml_file
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)
    cache_dir = args_local["cache_dir"]
    sam2_checkpoint_path = args_local["sam2_checkpoint_path"]
    
    logger.log(f"Generating settings | images for class: {gen_class} | dataset: {dataset} | with {images_per_real} images per real image | using {batch_size} batch size | and {n_img_per_cls} real images per class | use mask generation: {use_mask} | back: {back} | color: {color} | texture: {texture} | objsize: {objsize} | feasible: {feasible} | infeasible: {infeasible}.")
    
    #### GroudningSAM related ####
    detection_threshold = args.detection_threshold
    detector_id = args.detector_id
    segmenter_id = args.segmenter_id
    
    #### Dataloader related ####
    csv_path = args.csv_path
    subset = args.subset
    download = args.download

    # vlm filtering related
    use_vlm_filtering = args.use_vlm_filtering

    # set the saving path
    dataset_root, gen_save_root, real_train_root = create_out_dir(args)
    all_class_names = SUBSET_NAMES[dataset]

    train_loader, _, image_root = get_data_loader(
        real_train_data_dir=dataset_root,
        real_test_data_dir=dataset_root,
        metadata_dir=n_img_per_cls,
        dataset=dataset,
        bs=1,
        eval_bs=1,
        n_img_per_cls=n_img_per_cls,
        is_synth_train=False,
        is_real_shots=False,
        model_type=None,
        is_rand_aug=False,
        csv_path=csv_path,
        return_root=True,
        DOWNLOAD=download,
        subset=subset,
        OODGEN=True,
    )
        ## Initialize the needed classes
    grounding_mask_detector = GroundingMask(detector_id, segmenter_id, sam2_checkpoint_path, cache_dir,
                                            device=device, threshold=detection_threshold, use_sam2=True)
    

    if back:
        img_generator = BackGen(logger=logger, use_mask=use_mask, with_image=with_image,
            feasibility=feasible, cache_dir=cache_dir, use_compel=True,
            feasible_prompts=FEASIBLE_BACK, infeasible_prompts=INFEASIBLE_BACK, images_per_real=images_per_real, batch_size=batch_size, general_class=gen_class,
            dataset=dataset)
    elif color:
        img_generator = ColorGen(logger=logger, use_mask=use_mask, with_image=with_image,
            feasibility=feasible, cache_dir=cache_dir, use_compel=True,
            feasible_prompts=FEASIBLE_COLOR, infeasible_prompts=INFEASIBLE_COLOR, images_per_real=images_per_real, batch_size=batch_size, general_class=gen_class,
            dataset=dataset)
    elif texture:
        img_generator = TextureGen(logger=logger, use_mask=use_mask, with_image=with_image,
            feasibility=feasible, cache_dir=cache_dir, use_compel=True,
            feasible_prompts=FEASIBLE_TEXTURE, infeasible_prompts=INFEASIBLE_TEXTURE, images_per_real=images_per_real, batch_size=batch_size, general_class=gen_class,
            dataset=dataset, category='texture')
    else:
        raise RuntimeError("Invalid key") 
    
    if use_vlm_filtering:
        vqa_model_name = args.vqa_model_name
        vqa_scorer = VQAModelFiltering(vqa_model_name, load_in_Nbit=8, cache_dir=cache_dir, logger=logger)     
    
    # disable tqdm if you don't want to see the progress bar (especially for SLURM jobs)
    for idx, data in enumerate(tqdm(train_loader, disable=True)):
        image_tensor, label = data # check the dim here

        image_rgb_array = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255
        image_rgb_array = image_rgb_array.astype(np.uint8)
        
        this_class_ = all_class_names[label]
        file_this_stem = Path(image_root[idx]).stem
        # print(this_class_)
        # print('gen', args.gen_class)
        
        if this_class_ == gen_class:

            output_dir = os.path.join(gen_save_root, this_class_)
            os.makedirs(output_dir, exist_ok=True)

            real_train_dir_root_cls = os.path.join(real_train_root, this_class_)
            os.makedirs(real_train_dir_root_cls, exist_ok=True)

            # defines a sort of timeout for each class, if we can't generate an image for a class in 100 cycles, we move on with whatever the model generates
            cycles_spent_this_real = 0
            
            # save the shot images as back up
            real_image_path = os.path.join(real_train_dir_root_cls, file_this_stem + '.jpg')
            # print(real_image_path)
            if not os.path.exists(real_image_path):
                original_real_pil = Image.fromarray(image_rgb_array)
                original_real_pil.save(real_image_path)
                
            input_image, origianl_shape = general_preprocess(image_rgb_array.copy())
            
            if debug:
                input_image_pil = Image.fromarray(input_image)
                input_image_pil.save('input_image_debug.png')
            
            # get mask
            grounding_mask_detector.initialize(input_image)
            prompt_sam = grounding_mask_detector.get_prompt(dataset, this_class_)
            mask_array = grounding_mask_detector.get_mask(prompt_sam)
            
            if debug:
                mask_pil = Image.fromarray(mask_array)
                mask_pil.save('mask_debug.png')
            
            # useful in case of multiple parallel runs
            all_files = glob.glob(os.path.join(output_dir, f"{file_this_stem}*"))
            pattern = re.compile(rf"^{re.escape(file_this_stem)}(_|$)")
            matched_files = [f for f in all_files if pattern.search(os.path.basename(f))]
            generated_this_real = len(matched_files)
            
            img_generator.initialize_pipeline(input_image, mask_array, this_class_)
            
            while generated_this_real < images_per_real:
                cycles_spent_this_real += 1
                
                generated_images_batch = []
                
                for _ in range(REPEAT):
                    # generate the images
                    prompts_dict = img_generator.get_prompt(generated_this_real, this_class_)
                    input_images, mask_images, other_process_dict = img_generator.pre_process(preprocess_prompts=prompts_dict)
                    
                    if use_mask:
                        generate_images = img_generator.generate(prompts_dict, input_images, mask_images, other_process_dict)
                    else:
                        generate_images = img_generator.generate_no_mask()
                    generate_images = img_generator.post_process(generate_images)
                    # print('generate_images', generate_images[0].shape)
                    final_imgs = general_postprocess(generate_images, origianl_shape)
                    final_imgs = [Image.fromarray(final_img.astype(np.uint8)) for final_img in final_imgs]
                    generated_images_batch.extend(final_imgs)
                # generated_images_batch = group_outputs_for_batch_repeats(
                # generated_images_batch, batch_size, REPEAT, 1)
                
                # VLM filtering
                if use_vlm_filtering:
                    outputs_scores = []
                    for idx, image in enumerate(generated_images_batch):
                        # use vqa scorer to verify the generated images
                        generated_image, result_dict = None, None

                        question_answer_pairs = img_generator.get_vqa_question(prompts_dict)
                        try:
                            result_dict = vqa_scorer.get_tifa_score(question_answer_pairs, image, enable_logging=False)
                        except Exception as e:
                            logger.error(f"Error in get_tifa_score: {e}")
                            continue
                        if result_dict and result_dict.get("tifa_score") == 1:
                            outputs_scores.append(result_dict.get("tifa_score"))
                            break
                        logger.log(f"Generated {idx} image with TIFA score: {result_dict.get('tifa_score')}")
                        outputs_scores.append(result_dict.get("tifa_score"))
                    
                    top_score = max(outputs_scores)
                    if top_score >= 0.5 or cycles_spent_this_real > 2:
                        logger.log(f"VLM filtering: Image is most similar to {this_class_}")
                        random_name = random.randint(0, 1000000000)
                        out_path = os.path.join(output_dir, f"{file_this_stem}_{random_name}_{generated_this_real}.jpg")
                        process_and_save_image(generated_images_batch[outputs_scores.index(top_score)], out_path)
                        generated_this_real += 1
                else:
                    final_imgs = generated_images_batch[random.randint(0, len(generated_images_batch) - 1)]
                    random_name = random.randint(0, 1000000000)
                    out_path = os.path.join(output_dir, f"{file_this_stem}_{random_name}_{generated_this_real}.jpg")
                    process_and_save_image(generated_images_batch[0], out_path)
                    logger.log(f"Without vlm filtering to save {this_class_}, Image saved at: {out_path}")
                    generated_this_real += 1

                all_files = glob.glob(os.path.join(output_dir, f"{file_this_stem}*"))
                pattern = re.compile(rf"^{re.escape(file_this_stem)}(_|$)")
                matched_files = [f for f in all_files if pattern.search(os.path.basename(f))]
                generated_this_real = len(matched_files)
                
    end_time = time.time()
    execution_time = end_time - start_time
    logger.log(f"Process Time: {execution_time:.6f} s")



if __name__ == "__main__":
    debug_flag = False
    
    parser = argparse.ArgumentParser()
    ## data files related
    parser.add_argument("--dataset", type=str, default="oxford_pets")
    parser.add_argument("--datasets_root", type=str, default="/shared-network/yliu/projects/OODData/OODGen/data")
    parser.add_argument("--output_dir", type=str, default="/shared-network/yliu/projects/OODData/OODGen/output")
    parser.add_argument("--real_train_dir", type=str, default="/shared-network/yliu/projects/OODData/OODGen/real_train")
    ## some setting files
    parser.add_argument("--csv_path", type=str, default="/shared-network/yliu/projects/OODData/Finetune/artifacts")
    parser.add_argument("--yaml_file", type=str, default=None, required=True)
    ## model related
    parser.add_argument("--detector_id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--segmenter_id", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--vqa_model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--detection_threshold", type=float, default=0.3, help='detection threshold for groundingdino')

    ## generation diffusion related
    parser.add_argument("--guidance_scale", type=int, default=40.0)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=0.99)
    parser.add_argument("--n_img_per_cls", type=int, default=100, help='choosed real images per class')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--images_per_real", type=int, default=5)
    parser.add_argument("--general_class", type=str, default='airplane', help="the class to be generated")


    # Basic arguments
    parser.add_argument("--back", action="store_true", help="use background augmentation")
    parser.add_argument("--color", action="store_true", help="use color augmentation")
    parser.add_argument("--texture", action="store_true", help="use color augmentation")
    parser.add_argument("--objsize", action="store_true", help="use color augmentation")
    parser.add_argument("--feasiblity", action="store_true", help="generate feasible images")

    parser.add_argument("--no_mask", action="store_true", help="whether to use sdxl-inpainting, no mask means not to use inpainting model")
    parser.add_argument("--with_image", action="store_true", help="diffusion model with images input")
    parser.add_argument("--subset", action="store_true", help="use subset of the whole dataset")
    parser.add_argument("--download", action="store_true", help="use subset of the whole dataset")

    parser.add_argument("--gen_class", type=str, default=None, required=False, help="the class to be generated")
    # Mutually exclusive group for infeasible and feasible
    parser.add_argument("--use_vlm_filtering", action="store_true", help="use vlm to filter the generation images")

    args = parser.parse_args()

    # # Custom validation
    if (args.with_image):
        if not args.no_mask:
            raise RuntimeError("Error: -with_image could only be used when no_mask is given.")

    main(args, debug=debug_flag)
