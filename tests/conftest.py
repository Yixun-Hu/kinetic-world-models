import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# the FloWM reference file imports `lightning` only for the LightningModule base class
if "lightning" not in sys.modules:
    try:
        import lightning  # noqa: F401
    except ImportError:
        import torch.nn as nn
        stub = types.ModuleType("lightning")
        stub.LightningModule = nn.Module
        sys.modules["lightning"] = stub

# load the FloWM reference by file path (a system-wide `tests` package can shadow `tests.reference`)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("flowm_ref", os.path.join(ROOT, "tests", "reference", "flowm_models_2d.py"))
flowm_ref = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(flowm_ref)
sys.modules["flowm_ref"] = flowm_ref
