import copy
from typing import Dict, List, Optional, Tuple

import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, instantiate
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.structures import BitMasks, Boxes, BoxMode, Instances
from omegaconf import OmegaConf

__all__ = ["GeneralizedRCNN_with_proposals"]
@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_with_proposals(GeneralizedRCNN):


    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):

        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)        
        
        old_height , old_width = batched_inputs[0]['height'], batched_inputs[0]['width']
        img_height, img_width = images.image_sizes[0]
        detected_instances_new = [Instances((img_height, img_width))]
        
        # scale the boxes
        scale_x = img_width / old_width
        scale_y = img_height / old_height
        
        old_boxes = copy.deepcopy(detected_instances[0].pred_boxes.tensor)
        detected_instances_new[0].pred_boxes = copy.deepcopy(Boxes(old_boxes))
        detected_instances_new[0].pred_classes = copy.deepcopy(detected_instances[0].pred_classes)
        detected_instances_new[0].scores = copy.deepcopy(detected_instances[0].scores)
        detected_instances_new[0].pred_boxes.scale(scale_x, scale_y)
        
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                
            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances_new = [x.to(self.device) for x in detected_instances_new]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances_new)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
        
        
class DefaultPredictor_with_RPN(DefaultPredictor):
    
    def predict_with_proposals(self, original_image, bbox=None):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            if(bbox is not None):
                bbox[:, :4]        *= image.shape[1]/height
                box                 = Boxes(torch.as_tensor(bbox[:, :4]))
                a                   = Instances((height, width))
                a.proposal_boxes    = box
                a.objectness_logits = bbox[:, -1]*10
            else:
                box = Boxes(torch.as_tensor(np.array([[0,0,0,0]])))
                a = Instances((height, width))
                a.proposal_boxes    = box
                a.objectness_logits = [0]
            inputs = {"image": image, "height": height, "width": width, "proposals": a}
            
            predictions = self.model([inputs])[0]
            return predictions
        
    def predict_with_bbox(self, original_image, inst):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model.inference([inputs], detected_instances=[inst])[0]
            return predictions

class DefaultPredictor_Lazy:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from the weights specified in config (cfg.MODEL.WEIGHTS).
    2. Always take BGR image as the input and apply format conversion internally.
    3. Apply resizing defined by the config (`cfg.INPUT.{MIN,MAX}_SIZE_TEST`).
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            test dataset name in the config.


    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: a yacs CfgNode or a omegaconf dict object.
        """
        if isinstance(cfg, CfgNode):
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = build_model(self.cfg)
            if len(cfg.DATASETS.TEST):
                test_dataset = cfg.DATASETS.TEST[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)

            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )

            self.input_format = cfg.INPUT.FORMAT
        else:  # new LazyConfig
            self.cfg = cfg
            self.model = instantiate(cfg.model)
            test_dataset = OmegaConf.select(cfg, "dataloader.test.dataset.names", default=None)
            if isinstance(test_dataset, (list, tuple)):
                test_dataset = test_dataset[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(OmegaConf.select(cfg, "train.init_checkpoint", default=""))

            mapper = instantiate(cfg.dataloader.test.mapper)
            self.aug = mapper.augmentations
            self.input_format = mapper.image_format

        self.model.eval().cuda()
        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug(T.AugInput(original_image)).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
        
    def predict_with_bbox(self, original_image, inst):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug(T.AugInput(original_image)).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model.inference([inputs], detected_instances=[inst])[0]
            return predictions