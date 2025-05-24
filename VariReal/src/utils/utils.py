import os
import cv2
import numpy as np
from PIL import Image, ImageOps

import cv2
import numpy as np

def general_preprocess(image, new_size=1024):
    '''
    including padding/ resize the input images
        image:np.ndarray
        new_size: square number, int
    '''
    height, width = image.shape[:2]
    original_shape = [height, width]

    aspect_ratio = width / height
    new_width = new_size
    new_height = new_size

    if aspect_ratio != 1:
        if width > height:
            new_height = int(new_size / aspect_ratio)
        else:
            new_width = int(new_size * aspect_ratio)

    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    if len(image.shape) == 3:
        channels = image.shape[2]
        result = np.zeros((new_size, new_size, channels), dtype=image.dtype)
    else:
        # grey
        result = np.zeros((new_size, new_size), dtype=image.dtype)
        
    result[0:new_height, 0:new_width] = image_resized

    return result, original_shape


def general_postprocess(processed_images, original_shape, new_size=1024):
    '''
    including de-padding/ de-resize the input images
    args：
        processed_image (numpy.ndarray): shape new_size x new_size）。
        original_width (int): original width。
        original_height (int): original height。
        new_size (int): 1024。

    return：
        numpy.ndarray
    '''
    final_images = []
    
    original_height, original_width = original_shape
    aspect_ratio = original_width / original_height
    # print(original_height, original_width)

    new_width = new_size
    new_height = new_size

    if aspect_ratio != 1:
        if original_width > original_height:
            new_height = int(new_size / aspect_ratio)
        else:
            new_width = int(new_size * aspect_ratio)

    for processed_image in processed_images:
        cropped_image = processed_image[0:new_height, 0:new_width]
        restored_image = cv2.resize(cropped_image, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
        final_images.append(restored_image)
    return final_images

def get_generated_key(args):
    
    key = ''
    if args.no_mask:
        key += 'no_mask_'
        if args.with_image:
            key += 'with_image_'
        else:
            key += 'without_image_'
    else:
        key += 'mask_'
        
    if args.back:
        key += 'back_'
    elif args.color:
        key += 'color_'
    elif args.texture:
        key += 'texture_'
    elif args.objsize:
        key += 'objsize_'
    else:
        raise RuntimeError("Invalid key")
    
    if args.feasiblity:
        key += 'feasible'
    else:
        key += 'infeasible'
        
    return key

def save_image(images, out_path):
    if isinstance(images, list):
        for idx, image in enumerate(images):
            # file_name, ext = os.path.splitext(out_path)
            # image_out_path = f"{file_name}_{idx}{ext}"
            _save_single_image(images[0], out_path)
    else:
        _save_single_image(images, out_path)

def _save_single_image(image, out_path):
    if isinstance(image, Image.Image):
        # PIL style
        image.save(out_path)
        # print(f"Image saved using PIL at: {out_path}")
    elif isinstance(image, np.ndarray):
        # OpenCV style
        cv2.imwrite(out_path, image)
        # print(f"Image saved using OpenCV at: {out_path}")
    else:
        raise TypeError("Unsupported image format. Expected PIL Image or OpenCV ndarray.")

def process_and_save_image(final_imgs, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(final_imgs, out_path)


def get_dataset_name_for_template(dataset):
    dataset_name = {
        "oxford_pets": "pet ",
        "fgvc_aircraft": "aircraft ",
        "cars": "car ",
        "waterbirds": "bird ",
    }[dataset]
    return dataset_name


def create_out_dir(args):
    datasets_root = args.datasets_root
    dataset = args.dataset
    real_train_dir = args.real_train_dir
    output_root = args.output_dir
    
    dataset_root = f"{datasets_root}/{dataset}/"
    
    real_train_root = f"{real_train_dir}/{dataset}/"
    os.makedirs(real_train_root, exist_ok=True)
    
    if hasattr(args, 'no_mask') and not args.no_mask:
    
        generate_key = get_generated_key(args)
        
        output_root += f'/{dataset}/' 
        
        for key in generate_key.split('_'):
            output_root += key + '/'
            os.makedirs(output_root, exist_ok=True)
    
    return dataset_root, output_root, real_train_root

def group_outputs_for_batch_repeats(
    all_generated_outputs: list, batch_size: int, repeat_times: int, num_return_sequences: int
):
    grouped_outputs = []
    for idx in range(batch_size):
        for repeat_idx in range(repeat_times):
            start_index = idx * num_return_sequences + repeat_idx * (batch_size * num_return_sequences)
            grouped_outputs.extend(all_generated_outputs[start_index : start_index + num_return_sequences])
    return grouped_outputs