# -*- coding: utf-8 -*-
"""
Protocolo de parametros de los estimadores (``get_params`` / ``set_params`` / ``__repr__``), el mismo de
scikit-learn pero sin depender de el: un estimador de ARMxtend guarda en ``self.<nombre>`` cada argumento de su
constructor tal cual lo recibio, y con eso scikit-learn puede clonarlo (``sklearn.base.clone``), meterlo en un
``Pipeline`` o buscar sus parametros con ``GridSearchCV``.

@author Carlos Fernandez-Basso
"""
import inspect
from typing import Any, Dict, List


class ParamsMixin(object):
    """``get_params``, ``set_params`` y ``__repr__`` a partir de la firma de ``__init__``."""

    @classmethod
    def _param_names(cls) -> List[str]:
        signature = inspect.signature(cls.__init__)
        return sorted(name for name, parameter in signature.parameters.items()
                      if name != "self" and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Los parametros del constructor y su valor actual (`deep` se admite por compatibilidad: no hay anidados)."""
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> "ParamsMixin":
        """Cambia parametros del constructor; devuelve el propio estimador."""
        valid = self._param_names()
        for name, value in params.items():
            if name not in valid:
                raise ValueError("parametro no valido '{}' para {}; los validos son {}".format(
                    name, type(self).__name__, valid))
            setattr(self, name, value)
        return self

    def __sklearn_tags__(self) -> Any:
        """Etiquetas que pide scikit-learn >= 1.6 para usar el estimador en ``GridSearchCV`` o ``Pipeline``."""
        from sklearn.base import BaseEstimator       # solo se importa si scikit-learn esta instalado y lo pide

        return BaseEstimator.__sklearn_tags__(self)

    def __repr__(self) -> str:
        defaults = {name: parameter.default for name, parameter in inspect.signature(type(self).__init__).parameters.items()}
        changed = ["{}={!r}".format(name, value) for name, value in self.get_params().items()
                   if name not in defaults or defaults[name] is inspect.Parameter.empty or value != defaults[name]]
        return "{}({})".format(type(self).__name__, ", ".join(changed))
