#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STGNN Package for Epilepsy Seizure Onset Zone (SOZ) Localization

Spatio-Temporal Graph Neural Network implementation for identifying
the single channel that seizes first among all channels.
"""

from .stgnn_model import STGNN_SOZ_Locator

__all__ = ['STGNN_SOZ_Locator']


