"""
cloudcast: Tooling and infrastructure to enable cloud nowcasting.
"""

from __future__ import annotations

from importlib.metadata import version

from cloudcast import download

__all__ = ("__version__", "download")
__version__ = version(__name__)
