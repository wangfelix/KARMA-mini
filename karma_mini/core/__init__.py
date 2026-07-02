"""Core package: data structures, base agent, and the pipeline.

The pipeline is intentionally NOT imported here. Doing so would create an
import cycle (pipeline -> agents -> core.base_agent -> core package init ->
pipeline). Import it directly with ``from karma_mini.core.pipeline import
KARMAPipeline``.
"""
