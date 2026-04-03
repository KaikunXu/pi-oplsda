# src/piopls/utils.py
"""Utility module for the pi-oplsda project.

This module provides a suite of helper functions designed to decouple UI
interaction and I/O logic from the core OPLS-DA algorithm and permutation
testing models, maintaining the purity of the core algorithmic code.

Key features include:
1. Intelligent runtime environment detection (identifies Jupyter vs. CLI).
2. Dynamic progress bar adapter (utilizes tqdm in Jupyter and rich in CLI).
"""

from typing import Iterable, Any


def is_jupyter() -> bool:
    """Check if the code is running within a Jupyter Notebook/Lab environment.

    Returns:
        bool: True if running in a Jupyter environment
            (ZMQInteractiveShell), False otherwise.
    """
    try:
        from IPython.core.getipython import get_ipython
        ipython_instance = get_ipython()
        
        if ipython_instance is None:
            return False
            
        return ipython_instance.__class__.__name__ == 'ZMQInteractiveShell'
    except (NameError, ImportError):
        return False

def get_custom_progress(
    iterable: Iterable[Any],
    total: int,
    desc: str = "Permutation Test",
    color: str = "#7F7F7F",
    bar_length: int = 50
) -> Iterable[Any]:
    """Unified progress bar adapter for Jupyter (tqdm) and Terminal (rich).

    Automatically detects the runtime environment and returns an appropriate
    progress bar wrapper without altering the iteration logic.

    Args:
        iterable (Iterable[Any]): Iterable object (e.g., range or generator)
            to be wrapped.
        total (int): Total number of iterations for calculating percentage
            and ETA.
        desc (str, optional): Description text displayed on the left of the
            progress bar. Defaults to "Permutation Test".
        color (str, optional): Color of the progress bar (e.g., 'green',
            'blue'). Defaults to "green".
        bar_length (int, optional): Physical length/width of the progress
            bar. Defaults to 80.

    Returns:
        Iterable[Any]: A wrapped iterable that automatically updates the
            visual progress bar upon consumption.
    """
    if is_jupyter():
        # Jupyter environment: use tqdm for HTML rendering
        from tqdm import tqdm

        # Customize tqdm format to align visually with rich
        custom_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [ETA: {remaining}]"
        )

        return tqdm(
            iterable,
            total=total,
            desc=desc,
            ncols=bar_length,
            colour=color,
            bar_format=custom_format,
            leave=True
        )

    else:
        # Terminal environment: use rich for CLI rendering
        from rich.progress import (
            Progress,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
            TimeRemainingColumn
        )

        # Replicate tqdm's layout using rich components
        progress = Progress(
            TextColumn(f"[{color}][bold]{desc}:"),
            BarColumn(bar_width=bar_length, complete_style=color),
            TaskProgressColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
        )

        def rich_generator() -> Iterable[Any]:
            with progress:
                task = progress.add_task("task", total=total)
                for item in iterable:
                    yield item
                    progress.advance(task)

        return rich_generator()