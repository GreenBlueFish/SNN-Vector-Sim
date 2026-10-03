"""Application entry point for the SNN vector visualization.

Run with::

    python -m snn_vector_sim.app

This module wires the GUI together and starts the Qt event loop. The single
:class:`~snn_vector_sim.core.engine.SimulationEngine` is constructed *inside*
:class:`~snn_vector_sim.gui.main_window.MainWindow`, which also builds the four
plot widgets, the control panels, and the :class:`QtCore.QTimer` tick loop. So
the entry point's job is deliberately small: create the single
:class:`QtWidgets.QApplication`, construct and show the main window, and hand
control to the event loop.

Keeping the Qt-binding import and the ``QApplication`` lifecycle isolated here
mirrors the rest of the ``gui`` package, so the Qt-free ``core`` package stays
headless and testable while this module owns the one process-wide application
object.
"""

import sys

from PySide6 import QtWidgets

from snn_vector_sim.gui.main_window import MainWindow


def main() -> None:
    """Build the main window and run the Qt event loop.

    Creates the single process-wide :class:`QtWidgets.QApplication`, constructs
    the :class:`~snn_vector_sim.gui.main_window.MainWindow` (which builds the
    engine, plots, controls, and the tick timer), shows it, and blocks in
    ``app.exec()`` until the window is closed. The event loop's exit code is
    propagated through :func:`sys.exit`.
    """
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
