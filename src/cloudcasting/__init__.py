"""
cloudcasting: Tooling and infrastructure to enable cloud nowcasting.
"""

from __future__ import annotations

from importlib.metadata import version

from jaxtyping import install_import_hook

# Any module imported inside this `with` block, whose
# name begins with the specified string, will
# automatically have both `@jaxtyped` and the
# typechecker applied to all of their functions and
# dataclasses, meaning that they will be type-checked
# (and therefore shape-checked via jaxtyping) at runtime.
with install_import_hook("cloudcasting", "typeguard.typechecked"):
    from cloudcasting import models, validation

from cloudcasting import cli, dataset, download, metrics

__all__ = (
    "__version__",
    "download",
    "cli",
    "dataset",
    "models",
    "validation",
    "metrics",
)
__version__ = version(__name__)
