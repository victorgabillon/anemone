
from anemone.node_selector.uniform.uniform import UniformArgs
from anemone.node_selector.recurzipf.recur_zipf_base import RecurZipfBaseArgs
from anemone.node_selector.sequool.factory import SequoolArgs


AllNodeSelectorArgs = (
    RecurZipfBaseArgs | SequoolArgs | UniformArgs 
)