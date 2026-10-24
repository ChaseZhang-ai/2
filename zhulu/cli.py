"""Command-line entry point: ``python -m zhulu <command>``.

Commands mirror the pieces of the engine that are useful to poke at standalone:

* ``plan``      - build a workflow for a product tier and print its signature.
* ``schedule``  - read a jobs JSON and print the batching plan + speedup.
* ``dedup``     - fingerprint a folder of PGM/raw images and report duplicates.
* ``voc``       - summarise reviews from a text file (one per line).
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


