"""
Wrapper that applies tokenizer pickle compatibility patch before running a target script.

Patch:
  pickle.load — adds missing internal attrs to MicroTokenizer (transformers 4.46 compat)

Usage:
  python notebooks/_run_with_patches.py scripts/benchmark_bernburg.py
"""
import sys
import pickle

_orig_load = pickle.load
def _patched_load(f, *a, **k):
    obj = _orig_load(f, *a, **k)
    if "Tokenizer" in type(obj).__name__:
        for attr, val in [
            ("_added_tokens_encoder", {}),
            ("_added_tokens_decoder", {}),
            ("_special_tokens_map", {}),
        ]:
            if not hasattr(obj, attr):
                object.__setattr__(obj, attr, val)
    return obj
pickle.load = _patched_load

target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
with open(target, "r", encoding="utf-8") as f:
    code = compile(f.read(), target, "exec")
exec(code, {"__name__": "__main__", "__file__": target})
