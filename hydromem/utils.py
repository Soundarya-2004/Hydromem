"""Utility functions for HydroMem hydraulic pressure, decay, and sedimentation.

Implements mathematical models for memory pressure calculation, exponential evaporation decay,
dynamic recency calculations, and sedimentation threshold evaluations.
Pure Python standard library implementation with zero required heavy dependencies.
"""

import math
from typing import Optional


def calculate_pressure(text: str, emotion: float = 0.5, recency: float = 1.0) -> float:
    """Calculate hydraulic pressure of a memory item based on text length, emotion, and recency.

    Relevance is calculated as min(len(text) / 100, 1.0).
    Hydraulic pressure is a weighted sum:
        pressure = (relevance * 0.6) + (emotion * 0.3) + (recency * 0.1)
    The value is clipped to the [0.0, 1.0] range.

    Args:
        text: Memory textual content.
        emotion: Emotional valence/intensity rating in [0.0, 1.0]. Defaults to 0.5.
        recency: Recency factor in [0.0, 1.0]. Defaults to 1.0.

    Returns:
        float: Hydraulic pressure score between 0.0 and 1.0.
    """
    relevance = min(len(text) / 100.0, 1.0)
    raw_pressure = (relevance * 0.6) + (emotion * 0.3) + (recency * 0.1)
    return float(max(0.0, min(1.0, raw_pressure)))


def compute_dynamic_recency(created_at: float, current_time: Optional[float] = None) -> float:
    """Compute dynamic exponential recency factor based on memory age.

    Formula:
        age_seconds = current_time - created_at
        age_hours = age_seconds / 3600.0
        recency = exp(-age_hours / 24.0)

    Args:
        created_at: Epoch timestamp when memory was ingested.
        current_time: Reference timestamp (defaults to current time).

    Returns:
        float: Dynamic recency factor in (0.0, 1.0].
    """
    import time
    now = current_time if current_time is not None else time.time()
    age_seconds = max(0.0, now - created_at)
    age_hours = age_seconds / 3600.0
    recency = math.exp(-age_hours / 24.0)
    return float(max(0.0, min(1.0, recency)))


def compute_combined_score(
    relevance: float,
    recency: float,
    importance: float,
    weights: Optional[tuple] = None,
) -> float:
    """Combine relevance, dynamic recency, and intrinsic importance into a ranking score.

    Default score formula:
        0.5 * relevance + 0.3 * recency + 0.2 * importance

    Args:
        relevance: Search relevance score (keyword or vector similarity) in [0.0, 1.0].
        recency: Dynamic recency score in [0.0, 1.0].
        importance: Intrinsic memory importance / hydraulic pressure in [0.0, 1.0].
        weights: Optional tuple of (w_relevance, w_recency, w_importance).

    Returns:
        float: Combined ranking score.
    """
    if weights is None:
        w_rel, w_rec, w_imp = 0.5, 0.3, 0.2
    else:
        w_rel, w_rec, w_imp = weights
    score = (w_rel * relevance) + (w_rec * recency) + (w_imp * importance)
    return float(score)


def evaporation_strength(
    initial: float,
    lambda_val: float = 0.1,
    time_elapsed: float = 0.0,
) -> float:
    """Calculate remaining memory strength based on exponential hydraulic evaporation.

    Formula:
        strength = initial * exp(-lambda_val * time_elapsed)

    Args:
        initial: Initial pressure or strength value.
        lambda_val: Evaporation decay rate constant. Defaults to 0.1.
        time_elapsed: Elapsed time in seconds. Defaults to 0.0.

    Returns:
        float: Retained memory strength after elapsed time.
    """
    decay = math.exp(-lambda_val * max(0.0, time_elapsed))
    return float(initial * decay)


def is_sedimented(recall_count: int) -> bool:
    """Determine if a memory has sedimented into permanent storage.

    Sedimentation occurs when a memory is recalled more than 3 times (recall_count > 3).

    Args:
        recall_count: Total number of times the memory has been successfully recalled.

    Returns:
        bool: True if recall_count > 3, False otherwise.
    """
    return recall_count > 3
