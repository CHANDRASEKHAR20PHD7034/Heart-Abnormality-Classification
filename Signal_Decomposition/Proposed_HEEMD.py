import logging
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import numpy as np
import scipy
from scipy import signal
from scipy import integrate,interpolate
import matplotlib as plt
from matplotlib import *
import sys
from pylab import *
from tqdm import tqdm
from scipy.spatial.distance import hamming
from Code.Signal_Decomposition.utils import get_timeline
import warnings
warnings.filterwarnings('ignore')
class Proposed_HEEMD:
    """
    **Ensemble Empirical Mode Decomposition**
    Ensemble empirical mode decomposition (Proposed_HEEMD) [Wu2009]_
    is noise-assisted technique, which is meant to be more robust
    than simple Empirical Mode Decomposition (EMD). The robustness is
    checked by performing many decompositions on signals slightly
    perturbed from their initial position. In the grand average over
    all IMF results the noise will cancel each other out and the result
    is pure decomposition.

    """

    logger = logging.getLogger(__name__)

    noise_kinds_all = ["normal", "uniform"]

    def __init__(self, trials: int = 100, noise_width: float = 0.05, ext_EMD=None, parallel: bool = False, **kwargs):

        # Ensemble constants
        self.trials = trials
        self.noise_width = noise_width
        self.separate_trends = bool(kwargs.get("separate_trends", False))

        self.random = np.random.RandomState()
        self.noise_kind = kwargs.get("noise_kind", "normal")
        self.parallel = parallel
        self.processes = kwargs.get("processes")  # Optional[int]
        if self.processes is not None and not self.parallel:
            self.logger.warning("Passed value for process has no effect when `parallel` is False.")
        self.E_IMF = None  # Optional[np.ndarray]
        self.residue = None  # Optional[np.ndarray]
        self._all_imfs = {}

    def __call__(
        self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1, progress: bool = False
    ) -> np.ndarray:
        return self.Proposed_HEEMD(S, T=T, max_imf=max_imf, progress=progress)

    def __getstate__(self) -> Dict:
        self_dict = self.__dict__.copy()
        if "pool" in self_dict:
            del self_dict["pool"]
        return self_dict

    def generate_noise(self, scale: float, size: Union[int, Sequence[int]]) -> np.ndarray:

        if self.noise_kind == "normal":
            noise = self.random.normal(loc=0, scale=scale, size=size)
        elif self.noise_kind == "uniform":
            noise = self.random.uniform(low=-scale / 2, high=scale / 2, size=size)
        else:
            raise ValueError(
                "Unsupported noise kind. Please assigned `noise_kind` to be one of these: {0}".format(
                    str(self.noise_kinds_all)
                )
            )
        return noise

    def noise_seed(self, seed: int) -> None:
        """Set seed for noise generation."""
        self.random.seed(seed)

    def Proposed_HEEMD(
        self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1, progress: bool = False
    ) -> np.ndarray:

        if T is None:
            T = get_timeline(len(S), S.dtype)

        scale = self.noise_width * np.abs(np.max(S) - np.min(S))
        self._S = S
        self._T = T
        self._N = len(S)
        self._scale = scale
        self.max_imf = max_imf

        # For trial number of iterations perform EMD on a signal
        # with added white noise
        if self.parallel:
            map_pool = Pool(processes=self.processes)
        else:
            map_pool = map
        all_IMFs = map_pool(self._trial_update, range(self.trials))

        if self.parallel:
            map_pool.close()

        self._all_imfs = defaultdict(list)
        it = iter if not progress else lambda x: tqdm(x, desc="Proposed_HEEMD", total=self.trials)
        for (imfs, trend) in it(all_IMFs):

            # A bit of explanation here.
            # If the `trend` is not None, that means it was intentionally separated in the decomp process.
            # This might due to `separate_trends` flag which means that trends are summed up separately
            # and treated as the last component. Since `proto_eimfs` is a dict, that `-1` is treated literally
            # and **not** as the *last position*. We can then use that `-1` to always add it as the last pos
            # in the actual eIMF, which indicates the trend.
            if trend is not None:
                self._all_imfs[-1].append(trend)

            for imf_num, imf in enumerate(imfs):
                self._all_imfs[imf_num].append(imf)

        # Convert defaultdict back to dict and explicitly rename `-1` position to be {the last value} for consistency.
        self._all_imfs = dict(self._all_imfs)
        if -1 in self._all_imfs:
            self._all_imfs[len(self._all_imfs)] = self._all_imfs.pop(-1)

        for imf_num in self._all_imfs.keys():
            self._all_imfs[imf_num] = np.array(self._all_imfs[imf_num])

        self.E_IMF = self.ensemble_mean()
        self.residue = S - np.sum(self.E_IMF, axis=0)

        return self.E_IMF



    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:

        if self.E_IMF is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        return self.E_IMF, self.residue

    @property
    def all_imfs(self):

        return self._all_imfs

    def ensemble_count(self) -> List[int]:

        return [len(imfs) for imfs in self._all_imfs.values()]

    def ensemble_mean(self) -> np.ndarray:

        return np.array([imfs.mean(axis=0) for imfs in self._all_imfs.values()])

    def ensemble_std(self) -> np.ndarray:

        return np.array([imfs.std(axis=0) for imfs in self._all_imfs.values()])
import pylab as plt
import random
E_imfNo = np.zeros(50, dtype=int)
# Logging options
logging.basicConfig(level=logging.INFO)
# Proposed_HEEMD options
max_imf = -1
# Signal options
N = [500,456,600,765,498]
S = [4,9,3,2,7]
random_num_N = int(random.choice(N))
random_num_S = int(random.choice(S))
# printing random number
#N = 500
tMin, tMax = 0, random_num_S * np.pi
# Hamming distance calculation
# T (time interval of the signal)
T = np.linspace(tMin, tMax, random_num_N)
S_val = random_num_S * np.sin(4 * T) + 4 * np.cos(9 * T) + np.sin(8.11 * T + 1.2)
hamming_distance = hamming(T, S_val) / len(S_val)
print("Hamming distance value : ",hamming_distance)
# Prepare and run Proposed_HEEMD
Proposed_HEEMD = Proposed_HEEMD(trials=50)
Proposed_HEEMD.noise_seed(12345)
