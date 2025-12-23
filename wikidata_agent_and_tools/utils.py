# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 17:38:08 2025

@author: yurt3
"""

import json
import os
from typing import Dict


def load_wikidata_property_labels(
    filename: str = "wikidata_properties.json",
    auxiliary_dir: str = "auxiliary_files",
) -> Dict[str, str]:
    """
    Load Wikidata property labels from JSON.

    Search order:
    1. Current working directory
    2. auxiliary_files/ subdirectory

    Expected JSON format:
        {
            "P10": "video",
            "P101": "field of work",
            ...
        }

    Returns:
        dict[str, str]: { "Pxxx": "label", ... }

    Raises:
        FileNotFoundError: if file is not found in either location
        ValueError: if JSON structure is invalid
    """

    candidate_paths = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), auxiliary_dir, filename),
    ]

    file_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    if file_path is None:
        raise FileNotFoundError(
            f"{filename} not found in current directory or '{auxiliary_dir}/'"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Unexpected JSON structure: expected a dictionary of PID → label."
        )

    return data
