"""
.. include:: ../README.md

.. include:: ../examples/regression_example.md
.. include:: ../examples/calibration_example.md
"""
REGRESSION = 'regression'
CLASSIFICATION = 'classification'

from LaplacePINN.baselaplace import BaseLaplace, ParametricLaplace, FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace
from LaplacePINN.lllaplace import LLLaplace, FullLLLaplace, KronLLLaplace, DiagLLLaplace
from LaplacePINN.subnetlaplace import SubnetLaplace
from LaplacePINN.laplace import Laplace
from LaplacePINN.marglik_training import marglik_training

__all__ = ['Laplace',  # direct access to all Laplace classes via unified interface
           'BaseLaplace', 'ParametricLaplace',  # base-class and its (first-level) subclasses
           'FullLaplace', 'KronLaplace', 'DiagLaplace', 'LowRankLaplace',  # all-weights
           'LLLaplace',  # base-class last-layer
           'FullLLLaplace', 'KronLLLaplace', 'DiagLLLaplace',  # last-layer
           'SubnetLaplace',  # subnetwork
           'marglik_training']  # methods
