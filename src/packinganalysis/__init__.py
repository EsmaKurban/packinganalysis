"""
packinganalysis is a package for analysing structural properties of
packings of granular particles in 3D obtained by Molecular Dynamics/Discrete
Element simulations.
"""

__author__ = """Esma Kurban"""
__email__ = 'esma.kurban92@gmail.com'
__version__ = '0.1.0-dev'

from .analysis import PackingAnalysis
from .particle_properties import (read_shape_data, particle_volume,
                                  volume_com_moment)
