"""Lexos Package.

This file makes core modules and classes available at the package level, so you can do imports like:

    from lexos import KMeansCluster

Last Updated: July 2, 2025
Last Tested: July 2, 2025
"""

from .kmeans import KMeans

__all__ = ["KMeans"]
