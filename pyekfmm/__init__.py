#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Yangkang Chen (chenyk2016@gmail.com), 2022-Present 
:license:
    MIT License
"""

__version__ = "0.0.9"

from .fmm import eikonal
from .fmm import eikonal_surf
from .fmm import eikonal_rtp
from .fmm import eikonal_plane

from .fmmvti import eikonalvti
from .fmmvti import eikonalvti_surf
from .fmmvti import eikonalvti_rtp

from eikonalc import *
from eikonalvtic import *



from .stream import stream2d
from .stream import stream3d
from .stream import trimrays
from .stream import ray2d
from .stream import ray3d
from .stream import extract
from .stream import extracts
from .stream import extracts2
from .plot import plot3d








