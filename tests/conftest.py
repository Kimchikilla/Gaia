"""Shared pytest setup."""

# Keep torch first on Windows to avoid intermittent c10.dll initialization
# failures when other native libraries are imported before PyTorch.
import torch  # noqa: F401
