


from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union, Callable, Any
from math import sqrt
import copy
import itertools
import sympy as sp
from scipy.integrate import quad

TVectorType = TypeVar("TVectorType", bound='IVector')


class IVector(ABC, Generic[TVectorType]):
    """
        This class is an interface class to represent a vector, with defined inner product.
        This implementation assumes the scalars over the real numbers field.
        The scalar is represented by the float type.
    """

    @abstractmethod
    def __mul__(self, scalar: float) -> 'TVectorType':
        pass

    @abstractmethod
    def __add__(self, other: 'TVectorType') -> 'TVectorType':
        pass

    @staticmethod
    @abstractmethod
    def inner(first: 'TVectorType', second: 'TVectorType') -> float:
        pass

    def __rmul__(self, scalar: float) -> 'TVectorType':
        return self.__mul__(scalar)

    def __neg__(self) -> 'TVectorType':
        return self.__mul__(-1)

    def __sub__(self, other) -> 'TVectorType':
        return self + (-other)

    def __truediv__(self, scalar: float) -> 'TVectorType':
        return 1 / scalar * self

    def norm(self) -> float:
        return sqrt(self.inner(self, self))

    def get_normalized(self) -> TVectorType:
        return self / self.norm()


class FiniteVectorSpace:
    def __init__(self, *base: IVector):
        # TODO consider to add the linear independence of the base check or to calculate the dim here
        self.dim = len(base)
        self.base: list[IVector] = list(map(copy.deepcopy, base))
        self.VEC_CLASS = self.base[0].__class__

    def get_orthonormal_base(self) -> list[IVector]:
        orthonormal_base = list()
        orthonormal_base.append(self.base[0].get_normalized())
        zero_vec = self.get_zero_vector()
        for i, b_vec in enumerate(self.base[1:], start=1):
            orthonormal_base.append(
                (b_vec - sum([self.VEC_CLASS.inner(b_vec, a) * a for a in orthonormal_base],
                             zero_vec)).get_normalized())
        return orthonormal_base

    def get_zero_vector(self):
        return self.base[0] - self.base[0]

    def get_best_approximation(self, vec: IVector) -> (IVector, float):
        zero_vec = self.get_zero_vector()
        orthonormal_base = self.get_orthonormal_base()
        w_star = sum([self.VEC_CLASS.inner(vec, a) * a for a in orthonormal_base], zero_vec)
        min_dist = (vec - w_star).norm()
        return w_star, min_dist


class RNVec(IVector['RNVec']):
    def __init__(self, *coord: float):
        super().__init__()
        self.n = len(coord)
        self.coord = list(coord)

    def __mul__(self, scalar: float) -> 'RNVec':
        return RNVec(*[x * scalar for x in self.coord])

    def __add__(self, other: 'RNVec') -> 'RNVec':
        return RNVec(*[x + y for x, y in zip(self.coord, other.coord)])

    @staticmethod
    def inner(first: 'RNVec', second: 'RNVec') -> float:
        return sum([x * y for x, y in zip(first.coord, second.coord)])

    def __str__(self):
        return f'({", ".join(map(str, self.coord))})'


class SymbolicFunction(IVector['SymbolicFunction']):
    def __init__(self, func: Any, ver: Any):
        super().__init__()
        self.func = func
        self.ver = ver

    def __mul__(self, scalar: float) -> 'SymbolicFunction':
        return SymbolicFunction(scalar * self.func, self.ver)

    def __add__(self, other: 'SymbolicFunction') -> 'SymbolicFunction':
        return SymbolicFunction(self.func + other.func, self.ver)

    @staticmethod
    def inner(first: 'SymbolicFunction', second: 'SymbolicFunction') -> float:
        # numerical integration, L2
        result, _ = quad(sp.lambdify(first.ver, first.func * second.func, 'numpy'), 0, 1)
        return result


def main():
    #a = RNVec(-1, 3, 1, 1)
    #b = RNVec(6, -8, -2, -4)
    #c = RNVec(6, 3, 6, -3)
    #W = FiniteVectorSpace(a, b, c)
    #aprox, error = W.get_best_approximation(RNVec(1, -1, 1, -2))
    #print(aprox, error)

    x = sp.symbols('x')
    base = []
    base.append(SymbolicFunction(sp.Integer(1), x))
    for n in range(1, 4):
        base.append(SymbolicFunction(sp.exp(-x**2*n), x))
        #base.append(SymbolicFunction(sp.sin(2*sp.pi*n*x), x))

    #print('f(x) = ' + sp.latex(w_star.func))
    W = FiniteVectorSpace(*base)
    w_star, min_dist = W.get_best_approximation(SymbolicFunction(-sp.ln(x+1), x))
    print('f(x) = ' + sp.latex(w_star.func))
    print(min_dist)

if __name__ == '__main__':
    main()
