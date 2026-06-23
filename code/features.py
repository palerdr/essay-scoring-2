from dataclasses import dataclass
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

FEATURE_COLUMNS = ["essay_id", "full_text"]
TARGET = "full_score"