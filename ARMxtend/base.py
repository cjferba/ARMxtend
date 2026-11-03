# -*- coding: utf-8 -*-
"""
Parameter protocol of the estimators (``get_params`` / ``set_params`` / ``__repr__``), the same as scikit-learn's but
without depending on it: an ARMxtend estimator stores each constructor argument in ``self.<name>`` exactly as it was
received, and with that scikit-learn can clone it (``sklearn.base.clone``), put it in a ``Pipeline`` or search its
parameters with ``GridSearchCV``.

@author Carlos Fernandez-Basso
"""
import inspect
from typing import Any, Dict, List


class ParamsMixin(object):
    """``get_params``, ``set_params`` and ``__repr__`` from the signature of ``__init__``."""

    @classmethod
    def _param_names(cls) -> List[str]:
        signature = inspect.signature(cls.__init__)
        return sorted(name for name, parameter in signature.parameters.items()
                      if name != "self" and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """The constructor parameters and their current value (`deep` is accepted for compatibility: nothing is nested)."""
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> "ParamsMixin":
        """Change constructor parameters; returns the estimator itself."""
        valid = self._param_names()
        for name, value in params.items():
            if name not in valid:
                raise ValueError("invalid parameter '{}' for {}; the valid ones are {}".format(
                    name, type(self).__name__, valid))
            setattr(self, name, value)
        return self

    def __sklearn_tags__(self) -> Any:
        """Tags that scikit-learn >= 1.6 asks for to use the estimator in ``GridSearchCV`` or ``Pipeline``."""
        from sklearn.base import BaseEstimator       # only imported if scikit-learn is installed and asks for it

        return BaseEstimator.__sklearn_tags__(self)

    def __repr__(self) -> str:
        defaults = {name: parameter.default for name, parameter in inspect.signature(type(self).__init__).parameters.items()}
        changed = ["{}={!r}".format(name, value) for name, value in self.get_params().items()
                   if name not in defaults or defaults[name] is inspect.Parameter.empty or value != defaults[name]]
        return "{}({})".format(type(self).__name__, ", ".join(changed))
