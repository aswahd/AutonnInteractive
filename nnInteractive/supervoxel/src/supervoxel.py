import concurrent.futures

import cc3d
import numpy as np
import torch
from reader import BloscReaderWriter, NfitiReaderWriter
from sam2.sam2.build_sam import build_sam2_video_predictor
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SuperVoxelGenerator:
    def __init__(self, input_dir, output_dir: str, config: dict):
        """
        SuperVoxelGenerator class constructor. This class is responsible for generating supervoxels
        segmentation masks from a list of image paths.

        Parameters:
            input_dir (str): The folder containing the images to process.
            output_dir (str): The directory where the segmentation masks will be saved. output_dir / name_of_image
            config (dict): A dictionary containing the configuration parameters for the SuperVoxel generation.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_format = config["file_format"]

        self.reader_writer = {".nii.gz": NfitiReaderWriter(), ".b2nd": BloscReaderWriter()}[self.file_format]
        print(f"Using {self.reader_writer} to read and write files")

        self.config = config

        self.sam = sam_model_registry["vit_h"](checkpoint=self.config["sam1_checkpoint"]).to("cuda")
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=48,
            points_per_batch=256,
            pred_iou_thresh=0.85,
            stability_score_thresh=0.92,
            box_nms_thresh=0.6,
            crop_nms_thresh=0.6,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=192,
        )
        model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam2_predictor = build_sam2_video_predictor(model_cfg, ckpt_path=self.config["sam2_checkpoint"])

    def sam2_propagation(self, sam2_predictor_state, image_data: np.ndarray, masks: list[np.ndarray], slice_idx: int):
        """
        Propagate the masks using SAM2

        Parameters:
            image_data (np.ndarray): The image data to process. Shape: (z, y, x)
            masks (List[np.ndarray]): A list of masks to propagate. Shape: (z, y, x)
            slice_idx (int): The index of the slice which contains the masks
        """
        # Only 0/1 values are ever written (and the result is binarized to uint8 downstream),
        # so uint8 suffices — float64 would be 8x the RAM for a full-volume stack.
        propagated_masks = np.zeros((len(masks), *image_data.shape), dtype=np.uint8)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

            for obj_id, m in enumerate(masks):
                # add new prompts and instantly get the output on the same frame
                frame_idx, object_ids, _ = self.sam2_predictor.add_new_mask(
                    sam2_predictor_state, slice_idx, obj_id, m.astype(np.int8)
                )

            # propagate the prompts to get masklets throughout the video
            for frame_idx, object_ids, out_masks in self.sam2_predictor.propagate_in_video(sam2_predictor_state):
                for obj_n, out_mask in zip(object_ids, out_masks):
                    propagated_masks[obj_n, frame_idx] = (out_mask > 0).detach().cpu().numpy()[0].astype(np.uint8)

            # reset state to predict in other direction
            self.sam2_predictor.reset_state(sam2_predictor_state)

            # flip images order
            sam2_predictor_state["images"] = torch.flip(sam2_predictor_state["images"], dims=(0,))
            max_frame = sam2_predictor_state["images"].shape[0] - 1

            for obj_id, m in enumerate(masks):
                # add new prompts and instantly get the output on the same frame
                frame_idx, object_ids, _ = self.sam2_predictor.add_new_mask(
                    sam2_predictor_state, max_frame - slice_idx, obj_id, m.astype(np.int8)
                )

            # propagate the prompts to get masklets throughout the video
            for frame_idx, object_ids, out_masks in self.sam2_predictor.propagate_in_video(sam2_predictor_state):
                for obj_n, out_mask in zip(object_ids, out_masks):
                    propagated_masks[obj_n, max_frame - frame_idx] = (
                        (out_mask > 0).detach().cpu().numpy()[0].astype(np.uint8)
                    )

        return propagated_masks

    def remove_other_components(self, masks, frame_idx):
        """
        Remove components other than the one overlapping with the seed mask in the given frame

        Parameters:
            mask (np.ndarray): The mask to process. Shape: (n, z, y, x)
            frame_idx (int): The index of the frame
        """
        filtered_masks = []
        for m in masks:
            m_cc = cc3d.connected_components(m, connectivity=26, binary_image=True)

            # keep one random component among those that overlap with the seed mask
            filtered_components = np.unique(m_cc[frame_idx])
            filtered_components = filtered_components[filtered_components != 0]
            if len(filtered_components) != 0:
                final_component = np.random.choice(filtered_components)
                filtered_masks.append((m_cc == final_component).astype(np.uint8))
        filtered_masks = np.stack(filtered_masks, axis=0)
        return filtered_masks

    def sam_supervoxel(self, image_data: np.ndarray):
        """
        Generate the supervoxels segmentation masks using SAM and SAM2

        Parameters:
            image_data (np.ndarray): The image data to process. Shape: (z, y, x)
        """
        # Normalize between 0 to 1 using 95 percentile
        p5, p95 = np.percentile(image_data, [5, 95])
        image_data = (image_data - p5) / (p95 - p5)
        data_shape = image_data.shape

        # Sample random slice with Gaussian probability
        z_len = data_shape[0]
        slice_probabilitys = np.exp(-np.linspace(-1, 1, data_shape[0]) ** 2 * 2)
        slice_idx = np.random.choice(z_len, p=slice_probabilitys / slice_probabilitys.sum())

        def generate_sam_masks():
            return self.mask_generator.generate(image_data[slice_idx, ..., None].repeat(3, axis=2))

        def init_sam2_predictor():
            return self.sam2_predictor.init_state(image_data)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_sam_masks = executor.submit(generate_sam_masks)
            future_sam2_predictor = executor.submit(init_sam2_predictor)

            masks = future_sam_masks.result()
            sam2_predictor_state = future_sam2_predictor.result()

        # Pick n masks randomly
        selected_masks = (
            np.random.choice(masks, size=self.config["masks_per_image"], replace=False)
            if len(masks) > self.config["masks_per_image"]
            else masks
        )
        selected_masks = [m["segmentation"] for m in selected_masks]

        # Propagate the masks using SAM2
        propagated_masks = self.sam2_propagation(sam2_predictor_state, image_data, selected_masks, slice_idx)
        self.sam2_predictor.reset_state(sam2_predictor_state)

        # Remove other components
        propagated_masks = self.remove_other_components(propagated_masks, slice_idx)

        # Binarize the masks
        propagated_masks = (propagated_masks > 0).astype(np.uint8)

        return propagated_masks
