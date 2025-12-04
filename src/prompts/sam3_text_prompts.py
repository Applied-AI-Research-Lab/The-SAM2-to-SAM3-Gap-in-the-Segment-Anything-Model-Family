#!/usr/bin/env python3
"""
SAM3 Text Prompt Templates and Utilities

This module provides text prompt generation and management for SAM3's
concept-driven segmentation. Unlike SAM2's geometric prompts, these are
natural language descriptions that express semantic concepts, attributes,
and compositional reasoning.

Effective text prompts are critical for SAM3 performance, requiring careful
consideration of vocabulary, specificity, and semantic grounding.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TextPromptTemplate:
    """
    Template system for generating concept-driven text prompts.
    
    This class manages prompt composition, variation, and organization
    for systematic SAM3 evaluation. It supports attribute combination,
    negative examples, and compositional concepts.
    """
    
    def __init__(self):
        """Initialize prompt template system."""
        self.base_concepts: Dict[str, List[str]] = {}
        self.attributes: Dict[str, Dict[str, List[str]]] = {}
        self.negative_concepts: Dict[str, List[str]] = {}
        
        logger.info("Initialized text prompt template system")
    
    def register_concept(
        self,
        concept_name: str,
        variations: List[str],
    ):
        """
        Register a base concept with linguistic variations.
        
        Multiple phrasings help test SAM3's language understanding robustness.
        
        Args:
            concept_name: Canonical concept identifier
            variations: List of equivalent text descriptions
        """
        self.base_concepts[concept_name] = variations
        logger.debug(f"Registered concept '{concept_name}' with {len(variations)} variations")
    
    def register_attributes(
        self,
        concept_name: str,
        attribute_type: str,
        values: List[str],
    ):
        """
        Register attribute values for compositional concept generation.
        
        Attributes enable fine-grained semantic specification beyond basic
        object categories, testing SAM3's compositional understanding.
        
        Args:
            concept_name: Base concept to attach attributes to
            attribute_type: Attribute category (e.g., color, size, state)
            values: Possible values for this attribute
        """
        if concept_name not in self.attributes:
            self.attributes[concept_name] = {}
        
        self.attributes[concept_name][attribute_type] = values
        logger.debug(f"Registered {len(values)} values for {concept_name}.{attribute_type}")
    
    def register_negatives(
        self,
        concept_name: str,
        negative_examples: List[str],
    ):
        """
        Register negative concepts for contrastive prompting.
        
        Negative examples help distinguish similar concepts and test
        semantic precision in SAM3.
        
        Args:
            concept_name: Positive concept
            negative_examples: Concepts to explicitly exclude
        """
        self.negative_concepts[concept_name] = negative_examples
        logger.debug(f"Registered {len(negative_examples)} negatives for '{concept_name}'")
    
    def generate_basic_prompt(
        self,
        concept_name: str,
        variation_idx: int = 0,
    ) -> str:
        """
        Generate basic concept prompt without attributes.
        
        Args:
            concept_name: Concept to generate prompt for
            variation_idx: Which linguistic variation to use
            
        Returns:
            Text prompt string
        """
        if concept_name not in self.base_concepts:
            logger.warning(f"Unknown concept '{concept_name}', using as-is")
            return concept_name
        
        variations = self.base_concepts[concept_name]
        idx = variation_idx % len(variations)
        
        return variations[idx]
    
    def generate_attribute_prompt(
        self,
        concept_name: str,
        attributes: Dict[str, str],
    ) -> str:
        """
        Generate compositional prompt with attributes.
        
        Combines base concept with attribute specifications to create
        precise semantic descriptions.
        
        Args:
            concept_name: Base concept
            attributes: Dictionary mapping attribute types to values
            
        Returns:
            Compositional text prompt
        """
        base = self.generate_basic_prompt(concept_name)
        
        # Prepend attribute modifiers
        modifiers = []
        for attr_type, attr_value in attributes.items():
            if concept_name in self.attributes and attr_type in self.attributes[concept_name]:
                if attr_value in self.attributes[concept_name][attr_type]:
                    modifiers.append(attr_value)
                else:
                    logger.warning(f"Unknown attribute value '{attr_value}' for {concept_name}.{attr_type}")
            else:
                logger.warning(f"Unregistered attribute type '{attr_type}' for {concept_name}")
        
        if modifiers:
            prompt = " ".join(modifiers) + " " + base
        else:
            prompt = base
        
        logger.debug(f"Generated attribute prompt: '{prompt}'")
        return prompt
    
    def generate_all_attribute_combinations(
        self,
        concept_name: str,
    ) -> List[Tuple[str, Dict[str, str]]]:
        """
        Generate all valid attribute combinations for a concept.
        
        This creates exhaustive prompt sets for comprehensive evaluation
        of compositional understanding.
        
        Args:
            concept_name: Base concept
            
        Returns:
            List of (prompt, attributes) tuples
        """
        if concept_name not in self.attributes:
            logger.warning(f"No attributes registered for '{concept_name}'")
            return [(self.generate_basic_prompt(concept_name), {})]
        
        # Generate Cartesian product of all attribute values
        attr_types = list(self.attributes[concept_name].keys())
        attr_values_lists = [self.attributes[concept_name][t] for t in attr_types]
        
        combinations = []
        self._generate_combinations_recursive(
            concept_name,
            attr_types,
            attr_values_lists,
            {},
            0,
            combinations
        )
        
        logger.info(f"Generated {len(combinations)} attribute combinations for '{concept_name}'")
        return combinations
    
    def _generate_combinations_recursive(
        self,
        concept_name: str,
        attr_types: List[str],
        attr_values_lists: List[List[str]],
        current_attrs: Dict[str, str],
        depth: int,
        results: List[Tuple[str, Dict[str, str]]],
    ):
        """
        Recursive helper for generating attribute combinations.
        """
        if depth == len(attr_types):
            prompt = self.generate_attribute_prompt(concept_name, current_attrs)
            results.append((prompt, current_attrs.copy()))
            return
        
        attr_type = attr_types[depth]
        for attr_value in attr_values_lists[depth]:
            current_attrs[attr_type] = attr_value
            self._generate_combinations_recursive(
                concept_name,
                attr_types,
                attr_values_lists,
                current_attrs,
                depth + 1,
                results
            )
            del current_attrs[attr_type]
    
    def generate_contrastive_prompt(
        self,
        concept_name: str,
        negative_concept: Optional[str] = None,
    ) -> str:
        """
        Generate prompt with explicit negative concept exclusion.
        
        Contrastive prompts test SAM3's ability to distinguish between
        semantically similar concepts.
        
        Args:
            concept_name: Positive concept to segment
            negative_concept: Specific concept to exclude (uses registered negatives if None)
            
        Returns:
            Contrastive prompt string
        """
        positive = self.generate_basic_prompt(concept_name)
        
        if negative_concept is None:
            if concept_name in self.negative_concepts:
                negative = self.negative_concepts[concept_name][0]
            else:
                logger.warning(f"No negatives registered for '{concept_name}'")
                return positive
        else:
            negative = negative_concept
        
        prompt = f"{positive} but not {negative}"
        
        logger.debug(f"Generated contrastive prompt: '{prompt}'")
        return prompt
    
    def generate_count_prompt(
        self,
        concept_name: str,
        count: Optional[int] = None,
    ) -> str:
        """
        Generate prompt with quantity specification.
        
        Tests SAM3's numerical reasoning and instance counting capabilities.
        
        Args:
            concept_name: Object concept
            count: Specific count (None for "all")
            
        Returns:
            Count-aware prompt
        """
        base = self.generate_basic_prompt(concept_name)
        
        if count is None:
            prompt = f"all {base}"
        elif count == 1:
            prompt = f"one {base}"
        else:
            prompt = f"{count} {base}"
        
        logger.debug(f"Generated count prompt: '{prompt}'")
        return prompt
    
    def generate_spatial_prompt(
        self,
        concept_name: str,
        spatial_relation: str,
        reference_concept: str,
    ) -> str:
        """
        Generate prompt with spatial relationships.
        
        Tests SAM3's understanding of spatial reasoning beyond individual
        object recognition.
        
        Args:
            concept_name: Target concept
            spatial_relation: Relation type (e.g., "near", "above", "left of")
            reference_concept: Reference object
            
        Returns:
            Spatial reasoning prompt
        """
        target = self.generate_basic_prompt(concept_name)
        reference = self.generate_basic_prompt(reference_concept)
        
        prompt = f"{target} {spatial_relation} {reference}"
        
        logger.debug(f"Generated spatial prompt: '{prompt}'")
        return prompt


def create_mineapple_prompts() -> TextPromptTemplate:
    """
    Create prompt templates specifically for MineApple dataset evaluation.
    
    This configures concept-driven prompts relevant to apple orchard
    segmentation, including ripeness, health, and variety attributes.
    
    Returns:
        Configured TextPromptTemplate for MineApple experiments
    """
    template = TextPromptTemplate()
    
    # Register base apple concepts with variations
    template.register_concept(
        "apples",
        variations=[
            "apples",
            "apple fruits",
            "apple instances",
        ]
    )
    
    template.register_concept(
        "apple_clusters",
        variations=[
            "apple clusters",
            "groups of apples",
            "clustered apples",
        ]
    )
    
    template.register_concept(
        "individual_apples",
        variations=[
            "individual apples",
            "single apples",
            "isolated apple fruits",
        ]
    )
    
    # Register ripeness attributes
    template.register_attributes(
        "apples",
        "ripeness",
        values=[
            "ripe",
            "unripe",
            "overripe",
            "mature",
            "immature",
        ]
    )
    
    # Register color attributes
    template.register_attributes(
        "apples",
        "color",
        values=[
            "red",
            "green",
            "yellow",
            "partially red",
            "light green",
            "dark red",
        ]
    )
    
    # Register health status attributes
    template.register_attributes(
        "apples",
        "health",
        values=[
            "healthy",
            "damaged",
            "diseased",
            "pristine",
            "defective",
        ]
    )
    
    # Register size attributes
    template.register_attributes(
        "apples",
        "size",
        values=[
            "large",
            "small",
            "medium",
            "oversized",
            "undersized",
        ]
    )
    
    # Register negative concepts for contrastive learning
    template.register_negatives(
        "apples",
        negative_examples=[
            "leaves",
            "branches",
            "background",
            "sky",
            "trunk",
        ]
    )
    
    logger.info("Created MineApple-specific prompt templates")
    return template


def create_general_object_prompts() -> List[str]:
    """
    Create general-purpose object detection prompts for zero-shot evaluation.
    
    These test SAM3's open-vocabulary capabilities on concepts not seen
    during training.
    
    Returns:
        List of general object category prompts
    """
    prompts = [
        "fruits",
        "vegetation",
        "organic objects",
        "natural objects",
        "edible objects",
        "agricultural products",
        "round objects",
        "colored objects",
    ]
    
    logger.info(f"Created {len(prompts)} general object prompts")
    return prompts


def create_attribute_focused_prompts() -> Dict[str, List[str]]:
    """
    Create prompts focused on specific attribute reasoning.
    
    These isolate individual semantic attributes to evaluate SAM3's
    fine-grained understanding capabilities.
    
    Returns:
        Dictionary mapping attribute types to prompt lists
    """
    prompts = {
        "color": [
            "red objects",
            "green objects",
            "yellow objects",
            "objects with red color",
            "objects with green color",
        ],
        "state": [
            "ripe objects",
            "unripe objects",
            "mature objects",
            "immature objects",
        ],
        "quality": [
            "healthy objects",
            "damaged objects",
            "pristine objects",
            "defective objects",
        ],
        "size": [
            "large objects",
            "small objects",
            "medium-sized objects",
        ],
    }
    
    total = sum(len(v) for v in prompts.values())
    logger.info(f"Created {total} attribute-focused prompts across {len(prompts)} categories")
    
    return prompts


def create_compositional_prompts(
    base_concept: str,
    attributes: List[Dict[str, str]],
) -> List[str]:
    """
    Create compositional prompts combining multiple attributes.
    
    Tests SAM3's ability to understand complex multi-attribute concepts
    requiring compositional reasoning.
    
    Args:
        base_concept: Base object category
        attributes: List of attribute dictionaries to combine
        
    Returns:
        List of compositional prompt strings
    """
    prompts = []
    
    for attr_dict in attributes:
        modifiers = " ".join(attr_dict.values())
        prompt = f"{modifiers} {base_concept}"
        prompts.append(prompt)
    
    logger.info(f"Created {len(prompts)} compositional prompts")
    return prompts


def create_negation_prompts(
    positive_concepts: List[str],
    negative_concepts: List[str],
) -> List[str]:
    """
    Create prompts with explicit negation for contrastive evaluation.
    
    Tests SAM3's ability to use negative information to refine segmentation.
    
    Args:
        positive_concepts: Concepts to include
        negative_concepts: Concepts to explicitly exclude
        
    Returns:
        List of negation-based prompts
    """
    prompts = []
    
    for pos in positive_concepts:
        for neg in negative_concepts:
            prompts.append(f"{pos} but not {neg}")
            prompts.append(f"{pos} excluding {neg}")
    
    logger.info(f"Created {len(prompts)} negation prompts")
    return prompts


def augment_prompt_with_synonyms(
    base_prompt: str,
    synonym_mapping: Dict[str, List[str]],
) -> List[str]:
    """
    Generate prompt variations using synonym substitution.
    
    Tests robustness of SAM3's language understanding across different
    linguistic expressions of the same concept.
    
    Args:
        base_prompt: Original prompt text
        synonym_mapping: Dictionary mapping words to their synonyms
        
    Returns:
        List of augmented prompts with synonym substitutions
    """
    prompts = [base_prompt]
    
    words = base_prompt.split()
    for i, word in enumerate(words):
        if word in synonym_mapping:
            for synonym in synonym_mapping[word]:
                new_words = words.copy()
                new_words[i] = synonym
                prompts.append(" ".join(new_words))
    
    logger.debug(f"Augmented prompt to {len(prompts)} variations")
    return prompts


def validate_prompt_quality(
    prompt: str,
    min_words: int = 2,
    max_words: int = 15,
    forbidden_tokens: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """
    Validate prompt quality and provide feedback.
    
    Ensures prompts meet basic quality criteria for effective SAM3 evaluation.
    
    Args:
        prompt: Text prompt to validate
        min_words: Minimum word count
        max_words: Maximum word count
        forbidden_tokens: Set of tokens that should not appear
        
    Returns:
        Tuple of (is_valid, feedback_message)
    """
    words = prompt.split()
    
    if len(words) < min_words:
        return False, f"Prompt too short ({len(words)} < {min_words} words)"
    
    if len(words) > max_words:
        return False, f"Prompt too long ({len(words)} > {max_words} words)"
    
    if forbidden_tokens:
        found_forbidden = [w for w in words if w.lower() in forbidden_tokens]
        if found_forbidden:
            return False, f"Contains forbidden tokens: {found_forbidden}"
    
    if not any(c.isalpha() for c in prompt):
        return False, "Prompt contains no alphabetic characters"
    
    return True, "Valid prompt"
