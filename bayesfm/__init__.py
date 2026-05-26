import warnings

warnings.warn(
    "bayesfm is deprecated and will no longer be maintained. "
    "The same classes are now available in the 'empfin' package. "
    "Install it with `pip install empfin` and replace `import bayesfm` "
    "with `import empfin`.",
    FutureWarning,
    stacklevel=2,
)

from .bfm import BFM, BFMGLS, BFMOMIT
from .example_data import get_ffrf, get_ff5f, get_ff25p

__all__ = [
    'BFM',
    'BFMGLS',
    'BFMOMIT',
    'get_ff25p',
    'get_ffrf',
    'get_ff5f',
]
