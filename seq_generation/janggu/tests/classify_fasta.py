# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:03:31 2021

@author: Bill Sun
"""

import argparse
import os
import numpy as np
import pandas as pd
import pkg_resources

import tensorflow as tf

from janggu import Janggu
from janggu import Scorer
from janggu import inputlayer
from janggu import outputdense
from janggu.data import Array
from janggu.data import Bioseq
from janggu.layers import Complement
from janggu.layers import DnaConv2D
from janggu.layers import Reverse
from janggu.utils import ExportClustermap
from janggu.utils import ExportTsv