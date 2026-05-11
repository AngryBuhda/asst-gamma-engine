"""asst-gamma-engine — open-source quantitative engine for the ASST
Bitcoin treasury Gamma Flywheel research project.

Subpackages:
- feeds: external data sources (Tiingo, BGeometrics, FlashAlpha, Steady, chain parsing)
- compute: derived signals (banding, stochastics, iv_band, legacy suggestions)
- selector: SelectorOutput pipeline
- orchestration: pipeline_state, gap checks, integrity sweep, regime alerts
- exports: research-grade exports (master research, selector quant, BCI evidence)

See the project README for architectural context.
"""

__version__ = "0.1.0.dev0"
__all__ = ["__version__"]
