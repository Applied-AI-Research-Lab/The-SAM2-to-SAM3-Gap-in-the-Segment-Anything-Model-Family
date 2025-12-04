#!/usr/bin/env python3
"""
SAM3 Model Wrapper

This module provides a clean interface to the SAM3 model for concept-driven
segmentation tasks. Unlike SAM2, SAM3 operates through multimodal prompts
including natural language text, visual exemplars, and geometric hints,
enabling semantic understanding and open-vocabulary reasoning.

The wrapper handles vision-language fusion, concept grounding, and semantic
mask generation to support concept-level segmentation experiments.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    logging.warning("SAM3 not installed. Please install from GitHub after HF authentication")
    Sam3Processor = None


logger = logging.getLogger(__name__)


class SAM3Wrapper:
    """
    Wrapper class for SAM3 model providing concept-driven segmentation.
    
    This wrapper encapsulates SAM3's multimodal capabilities, handling text
    encoding, vision-language fusion, concept grounding, and semantic mask
    generation. SAM3 fundamentally differs from SAM2 by understanding concepts
    rather than just spatial regions, enabling open-vocabulary segmentation.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        bpe_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        compile_model: bool = False,
    ):
        """
        Initialize SAM3 model with multimodal architecture.
        
        Args:
            checkpoint_path: Path to SAM3 checkpoint (None uses HuggingFace default)
            bpe_path: Path to BPE tokenizer vocabulary file
            device: Device to run model on (cuda or cpu)
            confidence_threshold: Minimum confidence for concept detection
            compile_model: Whether to compile model with torch.compile
        """
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.compile_model = compile_model
        
        logger.info("Initializing SAM3 model with vision-language architecture")
        self._load_model()
    
    def _load_model(self):
        """
        Load and initialize the SAM3 model with its multimodal components.
        
        This method builds the complete SAM3 architecture including:
        - Vision encoder (450M parameters)
        - Text encoder (300M parameters)  
        - Fusion layers for cross-modal alignment
        - Concept-conditioned mask decoder
        """
        if Sam3Processor is None:
            raise ImportError("SAM3 not available. Install: pip install git+https://github.com/facebookresearch/sam3.git")
        
        # Build SAM3 image model with vision-language components
        model = build_sam3_image_model(
            bpe_path=self.bpe_path,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            load_from_HF=self.checkpoint_path is None,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=self.compile_model,
        )
        
        # Wrap in processor for simplified interface
        self.processor = Sam3Processor(
            model=model,
            resolution=1008,  # SAM3 uses 1008x1008 resolution
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )
        
        logger.info("SAM3 model loaded with multimodal capabilities")
    
    def set_image(self, image: Union[np.ndarray, Image.Image]) -> Dict:
        """
        Process and encode image for concept-driven segmentation.
        
        This method computes both visual embeddings and prepares the image
        for subsequent text-conditioned or exemplar-based segmentation.
        Unlike SAM2, these embeddings will be fused with language features.
        
        Args:
            image: Input image as numpy array (H, W, 3) or PIL Image
            
        Returns:
            Dictionary containing inference state with vision embeddings
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        logger.debug(f"Processing image size {image.size} for SAM3")
        
        # Set image and compute vision-language aligned features
        inference_state = self.processor.set_image(image)
        
        return inference_state
    
    def segment_with_text(
        self,
        inference_state: Dict,
        text_prompt: str,
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Perform concept-level segmentation using natural language text.
        
        This is SAM3's core capability: understanding semantic concepts
        expressed in text and segmenting all matching instances without
        requiring spatial prompts. The text is encoded, aligned with visual
        features, and used to condition mask generation.
        
        Args:
            inference_state: Image state from set_image
            text_prompt: Natural language concept description (e.g., "ripe apples")
            
        Returns:
            Dictionary containing:
                - masks: Binary masks for all detected concept instances
                - boxes: Bounding boxes for each instance
                - scores: Confidence scores for each detection
                - labels: Semantic labels for each mask
        """
        logger.info(f"Segmenting concept: '{text_prompt}'")
        
        # Encode text and perform vision-language fusion
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt,
        )
        
        return {
            "masks": output["masks"],
            "boxes": output["boxes"],
            "scores": output["scores"],
            "labels": [text_prompt] * len(output["masks"]),
        }
    
    def segment_multiple_concepts(
        self,
        inference_state: Dict,
        text_prompts: List[str],
    ) -> Dict[str, List]:
        """
        Segment multiple concepts in a single image using text prompts.
        
        This method demonstrates SAM3's ability to reason about multiple
        semantic concepts simultaneously, detecting and segmenting instances
        of each concept independently.
        
        Args:
            inference_state: Image state from set_image
            text_prompts: List of concept descriptions
            
        Returns:
            Dictionary mapping concepts to their segmentation results
        """
        all_results = {
            "masks": [],
            "boxes": [],
            "scores": [],
            "labels": [],
        }
        
        for prompt in text_prompts:
            result = self.segment_with_text(inference_state, prompt)
            
            all_results["masks"].extend(result["masks"])
            all_results["boxes"].extend(result["boxes"])
            all_results["scores"].extend(result["scores"])
            all_results["labels"].extend(result["labels"])
        
        logger.info(f"Segmented {len(text_prompts)} concepts, found {len(all_results['masks'])} instances")
        return all_results
    
    def segment_with_attributes(
        self,
        inference_state: Dict,
        object_name: str,
        attributes: Dict[str, List[str]],
    ) -> Dict[str, Dict]:
        """
        Perform attribute-aware segmentation using compositional concepts.
        
        This demonstrates SAM3's semantic reasoning: segmenting objects
        based on attribute values like ripeness, color, or health status,
        which requires true concept understanding.
        
        Args:
            inference_state: Image state
            object_name: Base object category (e.g., "apples")
            attributes: Dictionary of attribute types and values
                       e.g., {"ripeness": ["ripe", "unripe"], "color": ["red", "green"]}
            
        Returns:
            Nested dictionary organizing results by attribute categories
        """
        results_by_attribute = {}
        
        for attr_type, attr_values in attributes.items():
            results_by_attribute[attr_type] = {}
            
            for attr_value in attr_values:
                # Construct compositional text prompt
                prompt = f"{attr_value} {object_name}"
                
                result = self.segment_with_text(inference_state, prompt)
                results_by_attribute[attr_type][attr_value] = result
                
                logger.debug(f"Found {len(result['masks'])} instances of '{prompt}'")
        
        return results_by_attribute
    
    def segment_with_exemplar(
        self,
        inference_state: Dict,
        exemplar_image: Union[np.ndarray, Image.Image],
        exemplar_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Segment using visual exemplar showing the target concept.
        
        SAM3 supports exemplar-based prompting where a reference image
        demonstrates the desired concept, enabling segmentation through
        visual similarity rather than language.
        
        Args:
            inference_state: Image state
            exemplar_image: Reference image showing target concept
            exemplar_mask: Optional mask highlighting exemplar region
            
        Returns:
            Dictionary with masks and scores for matching instances
        """
        if isinstance(exemplar_image, np.ndarray):
            exemplar_image = Image.fromarray(exemplar_image)
        
        logger.info("Performing exemplar-based concept segmentation")
        
        # Process exemplar through vision encoder
        # Note: This is a simplified interface; actual implementation
        # may require model-specific exemplar handling
        output = self.processor.set_exemplar_prompt(
            state=inference_state,
            exemplar=exemplar_image,
            exemplar_mask=exemplar_mask,
        )
        
        return {
            "masks": output["masks"],
            "boxes": output["boxes"],
            "scores": output["scores"],
        }
    
    def segment_with_negative_prompts(
        self,
        inference_state: Dict,
        positive_prompt: str,
        negative_prompts: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Segment target concept while explicitly excluding negative concepts.
        
        This leverages SAM3's semantic understanding to distinguish between
        similar concepts by providing negative examples of what not to segment.
        
        Args:
            inference_state: Image state
            positive_prompt: Target concept to segment
            negative_prompts: List of concepts to exclude
            
        Returns:
            Dictionary with filtered segmentation results
        """
        logger.info(f"Segmenting '{positive_prompt}' excluding {negative_prompts}")
        
        # Get positive concept results
        positive_results = self.segment_with_text(inference_state, positive_prompt)
        
        # Get negative concept results for filtering
        negative_masks = []
        for neg_prompt in negative_prompts:
            neg_result = self.segment_with_text(inference_state, neg_prompt)
            negative_masks.extend(neg_result["masks"])
        
        # Filter positive results that don't overlap with negatives
        # This is a simplified implementation; production code would use
        # more sophisticated semantic filtering
        filtered_masks = []
        filtered_boxes = []
        filtered_scores = []
        
        for mask, box, score in zip(
            positive_results["masks"],
            positive_results["boxes"],
            positive_results["scores"]
        ):
            # Check overlap with negative masks
            overlaps = False
            for neg_mask in negative_masks:
                if np.sum(mask & neg_mask) > 0.3 * np.sum(mask):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_masks.append(mask)
                filtered_boxes.append(box)
                filtered_scores.append(score)
        
        logger.info(f"Filtered to {len(filtered_masks)} instances after excluding negatives")
        
        return {
            "masks": np.array(filtered_masks),
            "boxes": np.array(filtered_boxes),
            "scores": np.array(filtered_scores),
        }
    
    def get_concept_embeddings(
        self,
        text_prompts: List[str],
    ) -> np.ndarray:
        """
        Extract text embeddings for concept prompts.
        
        This method exposes the text encoder to retrieve semantic embeddings
        for analysis, visualization, or similarity computations.
        
        Args:
            text_prompts: List of text concepts to encode
            
        Returns:
            Array of shape (N, D) with text embeddings
        """
        embeddings = []
        
        for prompt in text_prompts:
            # Encode text through SAM3's language model
            embedding = self.processor.model.encode_text(prompt)
            embeddings.append(embedding.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def compute_vision_language_similarity(
        self,
        inference_state: Dict,
        text_prompt: str,
    ) -> np.ndarray:
        """
        Compute pixel-level vision-language similarity map.
        
        This visualization shows how well different image regions align
        with the text concept, revealing SAM3's semantic grounding.
        
        Args:
            inference_state: Image state with vision features
            text_prompt: Concept to compute similarity for
            
        Returns:
            2D similarity map showing concept alignment per pixel
        """
        # Extract vision and text embeddings
        vision_features = inference_state["backbone_out"]["vision_features"]
        text_embedding = self.get_concept_embeddings([text_prompt])[0]
        
        # Compute cosine similarity
        vision_flat = vision_features.reshape(-1, vision_features.shape[-1])
        similarity = np.dot(vision_flat, text_embedding)
        similarity = similarity.reshape(vision_features.shape[:2])
        
        logger.debug(f"Computed similarity map for '{text_prompt}'")
        return similarity
    
    def reset(self):
        """Reset processor state and clear cached features."""
        self.processor.reset()
        logger.debug("SAM3 processor state reset")
    
    def to(self, device: str):
        """
        Move model to specified device.
        
        Args:
            device: Target device (cuda or cpu)
        """
        self.device = device
        self.processor.model.to(device)
        logger.info(f"SAM3 model moved to {device}")


class SAM3VideoWrapper:
    """
    Wrapper for SAM3 video segmentation with concept tracking.
    
    This class extends SAM3 to video sequences, combining temporal tracking
    from SAM2 with concept-level understanding, enabling tracking of semantic
    concepts across frames.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        bpe_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize SAM3 video predictor.
        
        Args:
            checkpoint_path: Path to SAM3 checkpoint
            bpe_path: Path to BPE tokenizer
            device: Device for inference
        """
        logger.info("Initializing SAM3 video predictor")
        
        self.video_predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            device=device,
        )
        
        self.device = device
    
    def start_session(
        self,
        video_path: str,
    ) -> Dict:
        """
        Start a video segmentation session.
        
        Args:
            video_path: Path to video file or frame directory
            
        Returns:
            Response dictionary with session information
        """
        response = self.video_predictor.handle_request(
            request={
                "action": "start_session",
                "video_path": video_path,
            }
        )
        
        logger.info(f"Started video session: {response.get('session_id')}")
        return response
    
    def add_text_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text_prompt: str,
    ) -> Dict:
        """
        Add text prompt for concept tracking in video.
        
        Args:
            session_id: Video session identifier
            frame_idx: Frame to add prompt on
            text_prompt: Concept description
            
        Returns:
            Response with detected instances
        """
        response = self.video_predictor.handle_request(
            request={
                "action": "add_prompt",
                "session_id": session_id,
                "frame_idx": frame_idx,
                "prompt_type": "text",
                "prompt": text_prompt,
            }
        )
        
        return response
    
    def propagate_concept(
        self,
        session_id: str,
    ) -> Dict:
        """
        Propagate concept-level tracking across video frames.
        
        Args:
            session_id: Video session identifier
            
        Returns:
            Dictionary with per-frame segmentation results
        """
        response = self.video_predictor.handle_request(
            request={
                "action": "propagate",
                "session_id": session_id,
            }
        )
        
        return response
