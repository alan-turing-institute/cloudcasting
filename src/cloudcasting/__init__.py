"""
cloudcasting: Tooling and infrastructure to enable cloud nowcasting.
"""

from __future__ import annotations

from importlib.metadata import version

from cloudcasting import cli, dataset, download

__all__ = ("__version__", "download", "cli", "dataset")
__version__ = version(__name__)
