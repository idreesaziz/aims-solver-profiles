from functools import lru_cache
from pathlib import Path
from models.drift_pooling_runtime_v82 import Predictor

@lru_cache(maxsize=1)
def _get():return Predictor(Path(__file__).resolve().parent/'payload')

def predict(input,labeled=None):return float(_get().predict(input,labeled))
