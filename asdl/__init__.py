import os
inverse = os.environ.get('inverse')
if inverse == "cholesky":
    from .utils import *
    from .symmatrix import *
elif inverse == "lu":
    from .utils_lu import *
    from .symmatrix import *
elif inverse == "trsm":
    from .utils_trsm import *
    from .symmatrix_trsm import *
else:
    raise Exception(inverse)
from .utils import *
from .vector import *
from .matrices import *
from .operations import *
from .core import *
from .gradient import *
from .mvp import *
precision = os.environ.get('precision')
if precision == "std":
    from .grad_maker import *
elif precision == "bf":
    from .grad_maker_bf import *
elif precision == "bf_as":
    from .grad_maker_bf_as import *
elif precision == "fp":
    from .grad_maker_fp import *
elif precision == "fp_as":
    from .grad_maker_fp_as import *
else:
    raise Exception(precision)
from .fisher import *
from .hessian import *
from .precondition import *
from .kernel import *
from .counter import *


__version__ = '0.1.0'
