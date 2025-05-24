from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import torch.nn.functional as F
import requests
import numpy as np
from .result_utils import DetectionResult, BoundingBox

from transformers import AutoModelForMaskGeneration, AutoModelForImageSegmentation, AutoProcessor, AutoModelForZeroShotObjectDetection
from torchvision.transforms.functional import normalize
from diffusers.utils import load_image, make_image_grid
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from OODGen.sam2.sam2.build_sam import build_sam2
# from OODGen.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from VariReal.third_party.sam2.sam2.build_sam import build_sam2
from VariReal.third_party.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

# GroundingDetection: Handles zero-shot object detection and background removal
class GroundingDetection:
    """
    A class to perform zero-shot object detection using Grounding DINO and background removal using RMBG.
    """
    def __init__(self, detector_id: Optional[str] = None, device="cuda", threshold: float = 0.3, cache_dir:str =None):
        """
        Initialize detection model and RMBG segmentation model.
        
        Args:
            detector_id (Optional[str]): HF model ID for Grounding DINO.
            device (str): Device to run inference on.
            threshold (float): Threshold for grounding text matching.
            cache_dir (str): Path to cache HF models.
        """
        
        self.detector_id = detector_id
        self.device = device
        self.threshold = threshold
        self.cache_dir = cache_dir

        self.grounding_dino, self.processor = self.load_grounding_dino()
        self.rmbg_model = self.load_rmbg()

    def load_grounding_dino(self):
        """
        Load the Grounding DINO model and its processor.
        
        Returns:
            (grounding_model, processor): The model and processor instances.
        """
        detector_id = "IDEA-Research/grounding-dino-base"
        # object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
        # build grounding dino from huggingface
        processor = AutoProcessor.from_pretrained(detector_id, cache_dir=self.cache_dir)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id, cache_dir=self.cache_dir).to(self.device)
        return grounding_model, processor


    def load_rmbg(self):
        """
        Load the background removal model RMBG.
        
        Returns:
            model (torch.nn.Module): RMBG model loaded onto the specified device.
        """
        model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True, cache_dir=self.cache_dir)
        model.to(self.device)
        return model

    def run_rmbg(self, image: np.ndarray):
        """
        Run background removal on input image.
        
        Args:
            image (np.ndarray): Input image in HWC format.
        
        Returns:
            result_image (np.ndarray): Grayscale mask from RMBG model.
        """
        image = self._preprocess_image(image).to(self.device)
        # print('image shape:', image.shape)
        # orig_im_size = image.shape[0:2]
        # inference
        result = self.rmbg_model(image)
        # print('result shape:', result[0][0].shape)
        # post process
        result_image = self._postprocess_image(result[0][0])
        return result_image

    def _preprocess_image(self, image, model_input_size: list=1024) -> torch.Tensor:
        """
        Normalize and resize image for RMBG input.
        
        Args:
            image (np.ndarray): Input image.
        
        Returns:
            torch.Tensor: Preprocessed image.
        """
        if len(image.shape) < 3:
            image = image[:, :, np.newaxis]
        # orig_im_size=im.shape[0:2]
        im_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return image

    def _postprocess_image(self, result: torch.Tensor) -> np.ndarray:
        """
        Convert model output into an 8-bit grayscale mask.
        
        Args:
            result (torch.Tensor): Raw RMBG model output.
        
        Returns:
            np.ndarray: Normalized and scaled grayscale mask.
        """
        result = torch.squeeze(result, 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        # print('im_array shape:', im_array.shape)
        im_array = np.squeeze(im_array)
        return im_array

    def detect(self,
            image,
            labels: List[str],
            grounding_dino_detector=None,
            processor=None
    ) -> List[Dict[str, Any]]:
        """
        Detect objects described by text labels using Grounding DINO.
        
        Args:
            image: Input image (PIL or np.ndarray).
            labels (List[str]): Text descriptions.
            grounding_dino_detector: (Optional) detector instance.
            processor: (Optional) processor instance.
        
        Returns:
            List[Dict[str, Any]]: Detection results with boxes and scores.
        """
        if grounding_dino_detector is None or processor is None:
            grounding_dino_detector = self.grounding_dino
            processor = self.processor
        device = grounding_dino_detector.device
        labels = [label if label.endswith(".") else label + "." for label in labels]

        inputs = processor(images=image, text=labels, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_dino_detector(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=self.threshold,
            target_sizes=[image.shape[:2]]
        )

        return results


class SAM:
    """
    A wrapper to run segmentation using SAM or SAM2.
    """
    def __init__(self, segmenter_id: Optional[str] = None, device='cuda', use_sam2=True, sam2_checkpoint=None, cache_dir=None):
        '''
        Including load the SAM and SAM2 models
        Args:
            segmenter_id:
            device:
        '''
        self.segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

        self.use_sam2 = use_sam2

        if self.use_sam2 and sam2_checkpoint:
            self.sam2_checkpoint = sam2_checkpoint
        else:
            raise RuntimeError('You must give the corresponding sam2 checkpoint if using SAM2 model.')

        self.device = device
        self.cache_dir = cache_dir

        self.process_tuple = self.load()


    def load(self):
        if self.use_sam2:
            return self.load_sam2()
        else:
            return self.load_sam()

    def load_sam(self):
        segmentator = AutoModelForMaskGeneration.from_pretrained(self.segmenter_id, cache_dir=self.cache_dir).to(self.device)
        processor = AutoProcessor.from_pretrained(self.segmenter_id, cache_dir=self.cache_dir)
        return segmentator, processor

    def load_sam2(self):
        # device = "cuda:2" if torch.cuda.is_available() else "cpu"
        segmenter_id = self.segmenter_id if self.segmenter_id is not None else "sam2_hiera_l.yaml"

        # build SAM2 image predictor
        sam2_model = build_sam2(segmenter_id, self.sam2_checkpoint, device=self.device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)

        return sam2_predictor

    def _get_boxes(self, results: DetectionResult) -> List[List[List[float]]]:
        boxes = []
        # for result in results:
        xyxy = list(results[0]['boxes'].cpu().numpy()) # here we only get the box with the top score
        boxes.append(xyxy)

        return [boxes]

    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)
        # Extract the vertices of the contour
        polygon = largest_contour.reshape(-1, 2).tolist()

        return polygon

    def _polygon_to_mask(self, polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert a polygon to a segmentation mask.

        Args:
        - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        - image_shape (tuple): Shape of the image (height, width) for the mask.

        Returns:
        - np.ndarray: Segmentation mask with the polygon filled.
        """
        # Create an empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Convert polygon to an array of points
        pts = np.array(polygon, dtype=np.int32)

        # Fill the polygon with white color (255)
        cv2.fillPoly(mask, [pts], color=(255,))

        return mask

    def _refine_masks(self, masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = self._mask_to_polygon(mask)
                mask = self._polygon_to_mask(polygon, shape)
                masks[idx] = mask

        return masks

    def _refine_masks_sam2(self, masks: np.ndarray, polygon_refinement: bool = False) -> List[np.ndarray]:
        # masks = masks.squeeze(0)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = self._mask_to_polygon(mask)
                mask = self._polygon_to_mask(polygon, shape)
                masks[idx] = mask

        return masks

    def segment_sam(
            self,
            image,
            detection_results: List[Dict[str, Any]],
            polygon_refinement: bool = False,
            segmentator=None, processor=None
    ) -> List[DetectionResult]:
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        """
        if segmentator or processor is None:
            segmentator, processor = self.process_tuple[0], self.process_tuple[1]

        device = segmentator.device
        # boxes = get_boxes(detection_results)
        boxes = self._get_boxes(detection_results)

        if isinstance(image, Image.Image):
            image_width, image_height = image.size
        elif isinstance(image, np.ndarray):
            image_height, image_width = image.shape[:2]
        if not boxes.size:
            boxes = np.array([[0, 0, image_width, image_height]])  # 将 boxes 设置为整张图像的大小
            print('Could not detect the class, using rmbg as back up')
        inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

        outputs = segmentator(**inputs)
        masks = processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = self._refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result['mask'] = mask

        return detection_results


    def segment_sam2(
            self,
            image,
            detection_results: List[Dict[str, Any]],
            polygon_refinement: bool = False,
            sam2_predictor=None,
            debug=False
    ) -> List[DetectionResult]:
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        image : in HWC format if np.ndarray, or WHC format if PIL Imag
        returned mask : "CxHxW"
        """

        # boxes = get_boxes(detection_results)
        # boxes = get_boxes(detection_results)
        if sam2_predictor is None:
            sam2_predictor = self.process_tuple

        boxes = detection_results[0]["boxes"].cpu().numpy()

        if not boxes.size:
            # boxes = np.array([[0, 0, image_width, image_height]])  # 将 boxes 设置为整张图像的大小
            print('Could not detect the class, using rmbg as back up')
            detection_results[0]['mask'] = None
            return detection_results

        # build SAM2 image predictor
        # sam2_checkpoint = "/shared-network/yliu/projects/OODData/OODGen/sam2/checkpoints/sam2_hiera_large.pt"
        # model_cfg = "sam2_hiera_l.yaml"
        # sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        # sam2_predictor = SAM2ImagePredictor(sam2_model)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            if isinstance(image, Image.Image):
                input = np.array(image.convert("RGB"))
            elif isinstance(image, np.ndarray):
                input = image
            sam2_predictor.set_image(input)
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        if debug:
            fig, ax = plt.subplots()
            ax.imshow(image)
            for box in boxes:
                width = box[2] - box[0]
                height = box[3] - box[1]

                rect = patches.Rectangle((box[0], box[1]), width, height, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

            plt.axis('off')
            plt.savefig('test.jpg', bbox_inches='tight', pad_inches=0.0)
            plt.show()
            masks = masks
            mask_tmp = Image.fromarray(masks)
            mask_tmp.save('mask_debug.jpg')

        masks = self._refine_masks_sam2(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result['mask'] = mask

        return detection_results



class GroundingMask:
    """
    Main controller for grounding + segmentation.
    """
    GROUNDING_SAM_PROMPTS = ["A [CLASS]."]
    def __init__(self, detector_id, segmenter_id, sam2_checkpoint, cache_dir, device="cuda", threshold: float = 0.3, use_sam2=True):
        '''

        Args:
            image: np.array HWC
            detector_id:
            segmenter_id:
            device:
            threshold:
            use_sam2:
            sam2_checkpoint:
            cache_dir:
        '''
        self.image = None
        self.use_sam2 = use_sam2
        self.detection_model = GroundingDetection(detector_id=detector_id, device=device, threshold=threshold, cache_dir=cache_dir)
        self.segmentation_model = SAM(segmenter_id=segmenter_id, device=device, use_sam2=use_sam2, sam2_checkpoint=sam2_checkpoint, cache_dir=cache_dir)
    def initialize(self, image: np.ndarray):
        """Store the input image to be used for later detection/segmentation."""
        self.image = image
    
    def get_prompt(self, dataset, cur_class):
        """Generate the class-specific prompt based on dataset name."""
        grounding_sam_prompt = [
            self.GROUNDING_SAM_PROMPTS[0].replace('[CLASS]', str(self.get_dataset_name_for_template(dataset) + cur_class))]
        return grounding_sam_prompt

    def get_dataset_name_for_template(self, dataset):
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
            "waterbirds_nobias": "bird ",
        }[dataset]
        return dataset_name

    def get_mask(self, prompt):
        """
        Run full detection and segmentation pipeline.
        
        Args:
            prompt (List[str]): Text prompt for object to segment.
        
        Returns:
            np.ndarray: Binary segmentation mask.
        """
        detection_results = self.detection_model.detect(self.image, prompt)

        if self.use_sam2:
            segmentation_results = self.segmentation_model.segment_sam2(self.image, detection_results,
                                                                       polygon_refinement=True)
        else:
            segmentation_results = self.segmentation_model.segment_sam(self.image, detection_results, polygon_refinement=True)

        if segmentation_results[0].get('mask', None) is not None:
            mask = segmentation_results[0].get('mask', None)
        else:
            mask = self.detection_model.run_rmbg(self.image)
        # print(mask.shape)

        return mask
