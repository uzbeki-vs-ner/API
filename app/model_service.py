"""
NER Model service using GLiNER (Generalist and Lightweight model for Named Entity Recognition).
This is a windowed GLiNER inference service following the API contract.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from gliner import GLiNER

logger = logging.getLogger(__name__)

# Entity labels
LABELS = ["ORG", "NAME", "GEO"]

# Window parameters
WORD_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")
MAX_WORDS = 384
STRIDE = 128
THRESHOLD = 0.5


class NERModelService:
    """Service wrapper for GLiNER model."""
    
    def __init__(self, model_dir: str):
        """
        Initialize the NER model service.
        
        Args:
            model_dir: Path to the model directory
        """
        self.model_dir = Path(model_dir)
        self.device = self._resolve_device()
        self.model = None
        self.is_loaded = False
    
    def _resolve_device(self) -> str:
        """Determine the device to use."""
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU")
            return "cuda"
        else:
            logger.info("CUDA is not available, using CPU")
            return "cpu"
    
    def load(self):
        """Load the GLiNER model."""
        try:
            logger.info(f"Loading GLiNER model from {self.model_dir}")
            
            # Check if model directory exists
            if not self.model_dir.exists():
                raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")
            
            # Check for required files
            required_files = ['gliner_config.json', 'pytorch_model.bin']
            for file in required_files:
                if not (self.model_dir / file).exists():
                    # Also check for .safetensors alternative
                    if file == 'pytorch_model.bin' and (self.model_dir / 'model.safetensors').exists():
                        continue
                    logger.warning(f"Missing file: {file}")
            
            # Load model
            self.model = GLiNER.from_pretrained(
                str(self.model_dir),
                local_files_only=True
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"GLiNER model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load GLiNER model: {e}", exc_info=True)
            raise
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Perform NER on a single text using windowed GLiNER inference.
        
        Args:
            text: Input text for NER
            
        Returns:
            List of entities with label, start, and end positions
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")
        
        # Get window spans
        spans = word_window_spans(text)
        candidates = []
        
        # Process each window
        for window_start, window_end in spans:
            window_text = text[window_start:window_end]
            
            # Get predictions for this window
            try:
                entities = self.model.predict_entities(
                    window_text,
                    LABELS,
                    flat_ner=True,
                    threshold=THRESHOLD
                )
                
                # Adjust offsets for window position
                for entity in entities:
                    candidates.append({
                        "label": entity["label"],
                        "start": window_start + int(entity["start"]),
                        "end": window_start + int(entity["end"]),
                        "score": float(entity.get("score", 1.0))
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to process window [{window_start}:{window_end}]: {e}")
                continue
        
        # Merge and deduplicate entities from overlapping windows
        merged_entities = merge_window_entities(candidates)
        
        # Return only required fields (remove score)
        return [
            {
                "label": entity["label"],
                "start": entity["start"],
                "end": entity["end"]
            }
            for entity in merged_entities
        ]
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Perform NER on multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of entity lists for each text
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def split_words(text: str) -> List[Tuple[str, int, int]]:
    """Split text into words with positions."""
    return [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def word_window_spans(text: str) -> List[Tuple[int, int]]:
    """Generate window spans for long texts."""
    tokens = split_words(text)
    if not tokens:
        return []
    
    if len(tokens) <= MAX_WORDS:
        return [(0, len(text))]
    
    spans = []
    for start_idx in range(0, len(tokens), STRIDE):
        end_idx = min(start_idx + MAX_WORDS, len(tokens))
        spans.append((tokens[start_idx][1], tokens[end_idx - 1][2]))
        if end_idx >= len(tokens):
            break
    
    # Add tail window if needed
    tail_start = max(0, len(tokens) - MAX_WORDS)
    tail = (tokens[tail_start][1], tokens[-1][2])
    if spans[-1] != tail:
        spans.append(tail)
    
    return spans


def merge_window_entities(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge entities from overlapping windows, keeping highest scores."""
    best: Dict[Tuple[str, int, int], float] = {}
    
    for row in candidates:
        key = (row["label"], int(row["start"]), int(row["end"]))
        score = float(row.get("score", 1.0))
        if key not in best or score > best[key]:
            best[key] = score
    
    # Sort by score (descending), then by position
    ranked = sorted(
        best.items(), 
        key=lambda item: (-item[1], item[0][1], item[0][2], item[0][0])
    )
    
    # Remove overlapping entities
    kept = []
    occupied = []
    
    for (label, start, end), score in ranked:
        # Check for overlap with existing entities
        if any(end > a and start < b for a, b in occupied):
            continue
        
        kept.append({
            "label": label,
            "start": start,
            "end": end,
            "score": score
        })
        occupied.append((start, end))
    
    # Sort by position
    kept.sort(key=lambda row: (row["start"], row["end"], row["label"]))
    
    return kept
