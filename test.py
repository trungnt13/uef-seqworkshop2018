from __future__ import print_function, division, absolute_import

import numpy as np

from helpers import load_imdb_data, load_digit_data

imdb = load_imdb_data('/tmp', override=False)
digit = load_digit_data('/tmp', override=False)
