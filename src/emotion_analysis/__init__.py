"""Emotion analysis for African languages.

Targets span two datasets / two scripts:
  BRIGHTER (Latin): afr, swa, hau
  EthioEmo (Ge'ez): amh, tir
Both share the same 6-label multilabel schema.
"""

__version__ = "0.1.0"

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
TARGET_LANGUAGES = ["afr", "swa", "hau", "amh", "tir"]
