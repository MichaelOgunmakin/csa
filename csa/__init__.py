__version__ = "0.1.0"


def setup_jupyter() -> None:
    """Configure Plotly to render correctly in Jupyter and Colab notebooks."""
    import plotly.io as pio
    try:
        import google.colab  # noqa: F401
        pio.renderers.default = "colab"
    except ImportError:
        pio.renderers.default = "notebook"
