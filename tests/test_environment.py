"""Tests de entorno — verifican que el repositorio está bien configurado."""

from pathlib import Path


def test_dependencias_principales():
    """pandas y pytest deben estar instalados y ser importables."""
    import pandas  # noqa: F401
    import pytest  # noqa: F401


def test_estructura_del_proyecto():
    """El proyecto debe tener la estructura src layout esperada."""
    raiz = Path(__file__).resolve().parent.parent
    assert (raiz / "src" / "analytics").is_dir()
    assert (raiz / "pyproject.toml").is_file()
    assert (raiz / "tests").is_dir()


def test_paquete_importable():
    """El paquete analytics debe poder importarse."""
    import analytics  # noqa: F401
