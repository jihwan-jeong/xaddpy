"""Defines helper functions and classes for compatibility with SymEngine."""

import abc

from typing import Set

import symengine as sy
import symengine.lib.symengine_wrapper as core


class BaseVar(metaclass=abc.ABCMeta):
    """A wrapper for a SymEngine variable."""

    def __init__(self, var: sy.Symbol):
        assert isinstance(var, sy.Symbol), (
            f"Expected a SymEngine Symbol, got {type(var)}."
        )
        self._var = var
        self._is_Boolean = False

    @property
    def var(self):
        return self._var

    @property
    def is_Symbol(self):
        return True

    @property
    def is_symbol(self):
        return True

    @property
    def is_Boolean(self):
        return self._is_Boolean

    @is_Boolean.setter
    def is_Boolean(self, value: bool):
        self._is_Boolean = value

    def __getattr__(self, attr):
        return getattr(self._var, attr)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return repr(self.var)

    def __str__(self) -> str:
        return str(self.var)


class BooleanVar(BaseVar):
    """A wrapper for a symbolic Boolean variable."""

    @property
    def is_Boolean(self):
        return True

    def __eq__(self, other):
        if not isinstance(other, BooleanVar):
            return False
        return str(self.var) == str(other.var)

    def __hash__(self):
        return hash(str(self.var))

    @property
    def free_symbols(self) -> Set[BaseVar]:
        return {self}


class RandomVar(BaseVar):
    """A wrapper for a symbolic random variable."""

    def __init__(self, var: sy.Symbol):
        super().__init__(var)
        if str(var).startswith('Bernoulli'):
            self.is_Boolean = True

    @property
    def is_Random(self):
        return True

    def __eq__(self, other):
        if not isinstance(other, RandomVar):
            return False
        return str(self.var) == str(other.var)

    def __hash__(self):
        return hash(str(self.var))

    @property
    def free_symbols(self) -> Set[BaseVar]:
        return {self}
