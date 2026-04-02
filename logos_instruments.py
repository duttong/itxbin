"""Backward-compatibility shim — real module is in logosdata/logos_instruments.py"""
import importlib.util, os, sys

_real = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'logosdata', 'logos_instruments.py')
_spec = importlib.util.spec_from_file_location('logos_instruments', _real)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['logos_instruments'] = _mod
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('__')})
