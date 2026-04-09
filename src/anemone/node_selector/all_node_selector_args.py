"""Define the union of all node selector args here to avoid circular imports."""

from anemone.node_selector.linoo import LinooArgs
from anemone.node_selector.recurzipf.recur_zipf_base import RecurZipfBaseArgs
from anemone.node_selector.sequool.factory import SequoolArgs
from anemone.node_selector.uniform.uniform import UniformArgs

AllNodeSelectorArgs = LinooArgs | RecurZipfBaseArgs | SequoolArgs | UniformArgs
