"""
NER Model service that integrates with the baseline prediction code.
Uses the exact same inference pipeline as baseline/predict.py.
"""

import json
import logging
from typing import List, Dict, Any
import torch
from transformers import AutoModelForTokenClassification
from pathlib import Path

from baseline.common import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_STRIDE,
    TAGS,
    ENTITY_LABELS,
    decode_bio_tokens,
    load_fast_tokenizer,
    resolve_device,
    tokenize_windows,
    validate_window,
)

logger = logging.getLogger(__name__)


class NERModelService:
    """Service wrapper for the baseline NER model."""
    
    def __init__(self, model_dir: str):
        """
        Initialize the NER model service.
        
        Args:
            model_dir: Path to the model directory
        """
        self.model_dir = Path(model_dir)
        self.device = resolve_device("auto")
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.max_length = DEFAULT_MAX_LENGTH
        self.stride = DEFAULT_STRIDE
        self.is_loaded = False
        
    def load(self):
        """Load the model, tokenizer, and configuration."""
        try:
            logger.info(f"Loading model from {self.model_dir}")
            
            # Read baseline config if it exists
            self._load_baseline_config()
            
            # Load tokenizer
            self.tokenizer = load_fast_tokenizer(str(self.model_dir))
            validate_window(self.tokenizer, self.max_length, self.stride)
            
            # Load model
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_dir
            ).to(self.device)
            
            # Extract and validate labels
            self.id2label = self._extract_labels()
            
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_baseline_config(self):
        """Load baseline configuration if it exists."""
        config_path = self.model_dir / "baseline_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                if not isinstance(config, dict):
                    logger.warning(f"{config_path}: expected a JSON object")
                    return
                
                self.max_length = int(config.get("max_length", DEFAULT_MAX_LENGTH))
                self.stride = int(config.get("stride", DEFAULT_STRIDE))
                logger.info(f"Loaded config: max_length={self.max_length}, stride={self.stride}")
                
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"Failed to load baseline config: {e}")
                # Use defaults
                self.max_length = DEFAULT_MAX_LENGTH
                self.stride = DEFAULT_STRIDE
    
    def _extract_labels(self) -> Dict[int, str]:
        """Extract and validate BIO labels from model config."""
        labels = {
            int(index): str(label) 
            for index, label in self.model.config.id2label.items()
        }
        expected = set(TAGS)
        if set(labels.values()) != expected or set(labels) != set(range(len(TAGS))):
            raise ValueError(f"model labels must be exactly {list(TAGS)}")
        return labels
    
    @torch.inference_mode()
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Perform NER on a single text using the same pipeline as baseline.
        
        Args:
            text: Input text for NER
            
        Returns:
            List of entities with label, start, and end positions
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")
        
        # Tokenize text into overlapping windows
        windows = []
        for feature, offsets in tokenize_windows(
            self.tokenizer,
            text,
            max_length=self.max_length,
            stride=self.stride,
        ):
            windows.append((feature, offsets))
        
        # Predict and aggregate scores for overlapping tokens
        record_scores: Dict[tuple, tuple] = {}
        
        for feature, offsets in windows:
            # Prepare batch for single window
            batch = self.tokenizer.pad(
                [feature],
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            
            # Get probabilities
            probabilities = torch.softmax(
                self.model(**batch).logits.float(), 
                dim=-1
            ).cpu()
            
            # Aggregate scores for each token
            for token_index, (start, end) in enumerate(offsets):
                if start == end:  # Skip special tokens
                    continue
                
                key = (start, end)
                score = probabilities[0, token_index]
                
                if key in record_scores:
                    previous, count = record_scores[key]
                    record_scores[key] = (previous + score, count + 1)
                else:
                    record_scores[key] = (score.clone(), 1)
        
        # Decode BIO tokens to entity spans
        tagged_tokens = []
        for (start, end), (score_sum, count) in sorted(record_scores.items()):
            label_id = int((score_sum / count).argmax().item())
            tagged_tokens.append((start, end, self.id2label[label_id]))
        
        entities = decode_bio_tokens(tagged_tokens)
        
        # Validate and deduplicate entities (defensive, decode_bio_tokens should handle this)
        return self._validate_entities(entities, text)
    
    def _validate_entities(
        self, 
        entities: List[Dict[str, Any]], 
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Validate entities to ensure they meet contract requirements.
        
        Args:
            entities: List of entities from decode_bio_tokens
            text: Original text
            
        Returns:
            Validated list of entities
        """
        valid_entities = []
        seen = set()
        
        for entity in entities:
            label = entity["label"]
            start = entity["start"]
            end = entity["end"]
            
            # Check label is valid
            if label not in ENTITY_LABELS:
                continue
            
            # Check boundaries are valid
            if not (isinstance(start, int) and isinstance(end, int)):
                continue
            if not (0 <= start < end <= len(text)):
                continue
            
            # Check for duplicates
            key = (label, start, end)
            if key in seen:
                continue
            
            # Check text is not empty or whitespace only
            entity_text = text[start:end]
            if not entity_text.strip():
                continue
            
            seen.add(key)
            valid_entities.append({
                "label": label,
                "start": start,
                "end": end
            })
        
        # Sort by start position for consistency
        valid_entities.sort(key=lambda x: (x["start"], x["end"], x["label"]))
        
        return valid_entities
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Perform NER on multiple texts sequentially.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of entity lists for each text
        """
        return [self.predict(text) for text in texts]
