#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Yangkang Chen (chenyk2016@gmail.com), 2022-Present 
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

__version__ = "0.0.8"

from .fmm import eikonal
from .fmm import eikonal_surf
from .fmm import eikonal_rtp

from .fmmvti import eikonalvti
from .fmmvti import eikonalvti_surf
from .fmmvti import eikonalvti_rtp

from eikonalc import *
from eikonalvtic import *



from .stream import stream2d
from .stream import stream3d
from .stream import trimrays











