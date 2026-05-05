"""Built-in pipeline stages registered when the API starts.

Importing this package triggers ``@register("name")`` decorators in each
submodule. Add new stages by creating a module here and importing it from
``builtin_stages``.
"""

from __future__ import annotations

from . import builtin_stages

__all__ = ["builtin_stages"]
