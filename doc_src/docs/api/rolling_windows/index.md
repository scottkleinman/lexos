# Rolling Windows

The `rolling_windows` module provides classes for calculating and visualizing statistical frequencies of terms over sliding windows.

The main module is [rolling_windows](rolling_windows.md).

The `rolling_windows` module has three built-in calculator classes, [counts](calculators/counts.md), [averages](calculators/averages.md), and [ratios](calculators/ratios.md). Custom calculators should inherit from [base_plotter](calculators/base_calculator.md).

The `rolling_windows` module has two built-in plotter classes, [simple_plotter](plotters/simple_plotter.md) and [plotly_plotter](plotters/plotly_plotter.md). Custom plotters should inherit from [base_plotter](plotters/base_plotter.md).
