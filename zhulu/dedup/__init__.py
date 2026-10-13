"""Content de-duplication: visual fingerprints + semantic diversity.

Why this exists
---------------
Short-video platforms de-rank near-duplicate uploads. A generative content
pipeline that produces hundreds of clips from one product photo will trip that
filter within days unless the output is both *visually* and *semantically*
diverse. The two layers here run on the CPU control node before anything is
handed to the publishing queue.
"""

from .index import BKTree, DuplicateMatch, NearDuplicateIndex
from .phash import Fingerprint, average_hash, difference_hash, fingerprint, hamming, phash
from .semantic import SemanticDedup, SemanticScore, cosine

__all__ = [
    "BKTree",
    "DuplicateMatch",
    "Fingerprint",
    "NearDuplicateIndex",
    "SemanticDedup",
    "SemanticScore",
    "average_hash",
    "cosine",
    "difference_hash",
    "fingerprint",
    "hamming",
    "phash",
]
