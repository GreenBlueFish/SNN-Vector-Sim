"""Live plot widgets for the SNN vector visualization prototype.

This module holds the PyQtGraph-based plot widgets rendered in the main window.
Each widget is a self-contained Qt widget that reads simulation state from the
Qt-free :class:`~snn_vector_sim.core.engine.SimulationEngine` and refreshes its
visuals on demand via an ``update(engine)`` method called from the GUI timer
tick.

Widgets:

- :class:`NetworkDiagram` - the fixed two-input / one-output network graphic
  with live synaptic-weight labels (Req 8).
- :class:`PotentialPlot` - the three membrane-potential time series (Req 9).
- :class:`CurrentPlot` - the two input-current time series (Req 10).
- :class:`VectorPlot` - the Vector_N arrow and fading history trail in the
  ``N_1``-``N_2`` plane (Req 11).

Qt binding isolation
--------------------
The Qt binding import lives in exactly one place here (``from PySide6 import
...``) so swapping to another binding (e.g. PyQt5) is a one-line change. These
imports require PySide6 and pyqtgraph (see ``requirements.txt``); they are only
needed when the GUI actually runs, so a headless environment without them can
still import and unit-test the Qt-free ``core`` package.
"""

from PySide6 import QtCore, QtWidgets  # noqa: F401  (QtWidgets used by later classes)
import numpy as np
import pyqtgraph as pg


# ---------------------------------------------------------------------------
# Fixed layout coordinates for the network diagram (in plot/data coordinates).
# Inputs sit on the left, the output on the right, matching the design.
# ---------------------------------------------------------------------------
_INPUT1_POS = (0.0, 1.0)    # Input_Neuron 1 (top-left)
_INPUT2_POS = (0.0, -1.0)   # Input_Neuron 2 (bottom-left)
_OUTPUT_POS = (2.0, 0.0)    # Output_Neuron (right, centered)


def _midpoint(a, b):
    """Return the midpoint ``((ax+bx)/2, (ay+by)/2)`` of two ``(x, y)`` points."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


class NetworkDiagram(pg.GraphicsLayoutWidget):
    """Diagram of the fixed network with live synaptic-weight labels.

    Depicts the two Input_Neurons, the single Output_Neuron, and the two
    weighted connections from each input to the output (Req 8.1). Each edge
    carries a text label showing the current value of its synaptic weight
    (``w1`` on the Input_Neuron 1 -> Output edge, ``w2`` on the Input_Neuron 2
    -> Output edge), refreshed live from the engine (Req 8.2).

    The widget subclasses :class:`pyqtgraph.GraphicsLayoutWidget` so it is a
    drop-in ``QWidget`` for the main window layout. All graphics items are
    created once in ``__init__``; :meth:`update` only mutates existing items,
    so no items are allocated per frame.

    Node layout (data coordinates): Input_Neuron 1 at ``(0, 1)``, Input_Neuron 2
    at ``(0, -1)`` on the left, Output_Neuron at ``(2, 0)`` on the right.
    """

    # Visual constants.
    _NODE_SIZE = 42                    # node marker diameter (pixels)
    _INPUT_BRUSH = (80, 160, 255)      # input node fill (light blue)
    _OUTPUT_BRUSH = (255, 140, 80)     # output node fill (orange)
    _SPIKE_BRUSH = (255, 240, 120)     # flash color when a node spikes (yellow)
    _EDGE_PEN = pg.mkPen((200, 200, 200), width=2)
    _LABEL_COLOR = (230, 230, 230)

    def __init__(self, parent=None):
        """Build the static diagram: edges, nodes, and weight labels.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)

        # A single ViewBox holding all diagram items. Lock the aspect ratio and
        # disable mouse interaction so the fixed topology stays put and readable.
        self._view = self.addViewBox()
        self._view.setAspectLocked(True)
        self._view.setMouseEnabled(x=False, y=False)
        self._view.setMenuEnabled(False)
        # A little padding around the three nodes.
        self._view.setRange(
            xRange=(-0.6, 2.6), yRange=(-1.8, 1.8), padding=0,
        )

        # --- Edges (drawn first so nodes render on top of them). ---
        # Edge from Input_Neuron 1 to the Output_Neuron.
        self._edge1 = pg.PlotDataItem(
            x=[_INPUT1_POS[0], _OUTPUT_POS[0]],
            y=[_INPUT1_POS[1], _OUTPUT_POS[1]],
            pen=self._EDGE_PEN,
        )
        # Edge from Input_Neuron 2 to the Output_Neuron.
        self._edge2 = pg.PlotDataItem(
            x=[_INPUT2_POS[0], _OUTPUT_POS[0]],
            y=[_INPUT2_POS[1], _OUTPUT_POS[1]],
            pen=self._EDGE_PEN,
        )
        self._view.addItem(self._edge1)
        self._view.addItem(self._edge2)

        # --- Nodes: one scatter item holding the three node markers. ---
        # Index order is fixed: 0 = input1, 1 = input2, 2 = output.
        self._nodes = pg.ScatterPlotItem(
            x=[_INPUT1_POS[0], _INPUT2_POS[0], _OUTPUT_POS[0]],
            y=[_INPUT1_POS[1], _INPUT2_POS[1], _OUTPUT_POS[1]],
            size=self._NODE_SIZE,
            pen=pg.mkPen((30, 30, 30), width=1),
            brush=[
                pg.mkBrush(*self._INPUT_BRUSH),
                pg.mkBrush(*self._INPUT_BRUSH),
                pg.mkBrush(*self._OUTPUT_BRUSH),
            ],
            pxMode=True,
        )
        self._view.addItem(self._nodes)

        # --- Node name labels, anchored on each node. ---
        self._add_node_label("In 1", _INPUT1_POS)
        self._add_node_label("In 2", _INPUT2_POS)
        self._add_node_label("Out", _OUTPUT_POS)

        # --- Weight labels near each edge midpoint. ---
        mid1 = _midpoint(_INPUT1_POS, _OUTPUT_POS)
        mid2 = _midpoint(_INPUT2_POS, _OUTPUT_POS)
        self._w1_label = pg.TextItem(
            text="w1 = 1.00", color=self._LABEL_COLOR, anchor=(0.5, 1.0),
        )
        self._w1_label.setPos(mid1[0], mid1[1])
        self._w2_label = pg.TextItem(
            text="w2 = 1.00", color=self._LABEL_COLOR, anchor=(0.5, 0.0),
        )
        self._w2_label.setPos(mid2[0], mid2[1])
        self._view.addItem(self._w1_label)
        self._view.addItem(self._w2_label)

    def _add_node_label(self, text, pos):
        """Create a centered :class:`pyqtgraph.TextItem` label on a node.

        Args:
            text: Short label text (e.g. ``"In 1"``).
            pos: ``(x, y)`` data-coordinate position of the node.
        """
        label = pg.TextItem(text=text, color=self._LABEL_COLOR, anchor=(0.5, 0.5))
        label.setPos(pos[0], pos[1])
        self._view.addItem(label)

    def update(self, engine):
        """Refresh the live weight labels (and optional spike flashes).

        Reads the current synaptic weights ``engine.w1`` and ``engine.w2`` and
        rewrites the two edge labels so the diagram always shows the live
        values (Req 8.2). As an optional touch, each node is briefly tinted when
        its neuron spiked on the last step (``engine.input1.spiked`` etc.),
        giving a lightweight visual pulse; this is purely cosmetic.

        Args:
            engine: The :class:`~snn_vector_sim.core.engine.SimulationEngine`
                whose ``w1``, ``w2``, ``input1``, ``input2``, and ``output`` are
                read. No engine state is mutated.
        """
        # Live weight labels (Req 8.2).
        self._w1_label.setText(f"w1 = {engine.w1:.2f}")
        self._w2_label.setText(f"w2 = {engine.w2:.2f}")

        # Optional cosmetic spike flash: tint a node yellow on the step it fired.
        in1_brush = self._SPIKE_BRUSH if engine.input1.spiked else self._INPUT_BRUSH
        in2_brush = self._SPIKE_BRUSH if engine.input2.spiked else self._INPUT_BRUSH
        out_brush = self._SPIKE_BRUSH if engine.output.spiked else self._OUTPUT_BRUSH
        self._nodes.setBrush([
            pg.mkBrush(*in1_brush),
            pg.mkBrush(*in2_brush),
            pg.mkBrush(*out_brush),
        ])


class PotentialPlot(pg.PlotWidget):
    """Live time-series of the three membrane potentials (Req 9).

    Shows one curve per neuron -- Input_Neuron 1, Input_Neuron 2, and the
    Output_Neuron -- on a shared ``time (ms)`` vs ``V (mV)`` axis. The three
    :class:`pyqtgraph.PlotDataItem` curves are created once in ``__init__`` with
    distinct colors and a legend; :meth:`update` only calls ``setData`` on the
    existing curves so nothing is allocated per frame (Req 9.1, 9.2).

    The widget subclasses :class:`pyqtgraph.PlotWidget`, so it drops straight
    into the main-window layout as a ``QWidget``.
    """

    # Distinct curve colors (match the network-diagram node palette where it
    # helps: inputs in blue-ish tones, output in orange).
    _IN1_PEN = pg.mkPen((80, 160, 255), width=2)    # Input_Neuron 1 (light blue)
    _IN2_PEN = pg.mkPen((120, 220, 160), width=2)   # Input_Neuron 2 (green)
    _OUT_PEN = pg.mkPen((255, 140, 80), width=2)     # Output_Neuron (orange)

    def __init__(self, parent=None):
        """Build the three potential curves, legend, and axis labels.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)

        self.setLabel("bottom", "time (ms)")
        self.setLabel("left", "V (mV)")
        self.addLegend()
        self.showGrid(x=True, y=True, alpha=0.3)

        # Create each curve once; update() reuses them via setData (Req 9.2).
        self._curve_in1 = self.plot(pen=self._IN1_PEN, name="In 1")
        self._curve_in2 = self.plot(pen=self._IN2_PEN, name="In 2")
        self._curve_out = self.plot(pen=self._OUT_PEN, name="Out")

    def update(self, engine):
        """Refresh the three potential curves from the engine history.

        Reads ``engine.history.potentials()`` -- ``(time, v_in1, v_in2,
        v_out)`` NumPy arrays -- and pushes each into its curve via ``setData``.
        Empty buffers yield empty arrays, which ``setData`` handles gracefully
        (the curves simply draw nothing).

        Args:
            engine: The :class:`~snn_vector_sim.core.engine.SimulationEngine`
                whose ``history`` is read. No engine state is mutated.
        """
        t, v_in1, v_in2, v_out = engine.history.potentials()
        self._curve_in1.setData(t, v_in1)
        self._curve_in2.setData(t, v_in2)
        self._curve_out.setData(t, v_out)


class CurrentPlot(pg.PlotWidget):
    """Live time-series of the two injected input currents (Req 10).

    Shows one curve per input current -- ``I1`` and ``I2`` -- on a shared
    ``time (ms)`` vs ``I (nA)`` axis. The two
    :class:`pyqtgraph.PlotDataItem` curves are created once in ``__init__`` with
    distinct colors and a legend; :meth:`update` only calls ``setData`` on the
    existing curves so nothing is allocated per frame (Req 10.1, 10.2).

    The widget subclasses :class:`pyqtgraph.PlotWidget`, so it drops straight
    into the main-window layout as a ``QWidget``.
    """

    _I1_PEN = pg.mkPen((80, 160, 255), width=2)    # current into Input_Neuron 1
    _I2_PEN = pg.mkPen((120, 220, 160), width=2)   # current into Input_Neuron 2

    def __init__(self, parent=None):
        """Build the two current curves, legend, and axis labels.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)

        self.setLabel("bottom", "time (ms)")
        self.setLabel("left", "I (nA)")
        self.addLegend()
        self.showGrid(x=True, y=True, alpha=0.3)

        # Create each curve once; update() reuses them via setData (Req 10.2).
        self._curve_i1 = self.plot(pen=self._I1_PEN, name="I1")
        self._curve_i2 = self.plot(pen=self._I2_PEN, name="I2")

    def update(self, engine):
        """Refresh the two current curves from the engine history.

        Reads ``engine.history.currents()`` -- ``(time, I1, I2)`` NumPy arrays
        -- and pushes each into its curve via ``setData``. Empty buffers yield
        empty arrays, which ``setData`` handles gracefully.

        Args:
            engine: The :class:`~snn_vector_sim.core.engine.SimulationEngine`
                whose ``history`` is read. No engine state is mutated.
        """
        t, i1, i2 = engine.history.currents()
        self._curve_i1.setData(t, i1)
        self._curve_i2.setData(t, i2)


class VectorPlot(pg.PlotWidget):
    """Vector_N as a live cyan arrow with a fading red trail line (Req 11).

    Draws the central vector ``N`` in the ``N_1``-``N_2`` plane as a cyan arrow
    from the origin ``(0, 0)`` to the current tip ``(n1, n2)`` (Req 11.1). The
    tip's recent path is drawn as a thin **red line** that follows the tip and
    fades out with age (Req 11.2); how fast it fades is controlled live via
    :meth:`set_trail_fade`. Everything refreshes each tick from the engine
    (Req 11.3).

    The arrow is two cooperating items: a :class:`pyqtgraph.PlotDataItem` shaft
    from the origin to the tip and a :class:`pyqtgraph.ArrowItem` head at the
    tip. The trail is a fixed stack of :attr:`_TRAIL_BANDS` thin-line
    :class:`pyqtgraph.PlotDataItem` "bands"; each band covers a contiguous slice
    of the trail (oldest to newest) and is drawn at its own pen alpha, so the
    single visible line fades from bright near the tip to transparent at its
    oldest end. Using a handful of bands (rather than one segment per point)
    keeps the per-frame redraw cheap.

    All graphics items are created once in ``__init__``; :meth:`update` only
    mutates them (``setData`` / ``setPos`` / pen alpha), so no plot items are
    allocated per frame.

    The view uses a fixed, symmetric range with a locked aspect ratio so the
    vector's direction and length read truthfully rather than being distorted by
    independent axis scaling.
    """

    # Arrow color (bright cyan) and trail color (red, as requested).
    _ARROW_COLOR = (0, 220, 255)
    _TRAIL_COLOR = (255, 40, 40)

    # Symmetric default half-extent of the view in N units. The vector and
    # trail typically live within a few weight-impulses of the origin.
    _RANGE = 5.0

    # The trail is drawn as a thin line split into this many age "bands", each
    # a single PlotDataItem with one pen alpha. Older bands are more
    # transparent, giving a smooth fade cheaply (a few items instead of one
    # per segment). More bands = smoother gradient, slightly more cost.
    _TRAIL_BANDS = 16

    # Trail line thickness in pixels (thin, as requested).
    _TRAIL_WIDTH = 1.5

    # Brightest alpha (newest part of the trail, nearest the tip).
    _ALPHA_MAX = 230

    def __init__(self, parent=None):
        """Build the arrow (line + head), the banded trail line, and the axes.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)

        self.setLabel("bottom", "N_1")
        self.setLabel("left", "N_2")
        self.showGrid(x=True, y=True, alpha=0.3)

        # Lock aspect ratio so the vector direction is not distorted, and use a
        # fixed symmetric range centered on the origin.
        self.setAspectLocked(True)
        self.setRange(
            xRange=(-self._RANGE, self._RANGE),
            yRange=(-self._RANGE, self._RANGE),
            padding=0,
        )

        # Trail fade strength in [0, 1]. Higher = the trail fades to invisible
        # faster as it ages (a shorter-looking tail); lower = the tail lingers
        # longer. Settable live via :meth:`set_trail_fade`.
        self._trail_fade = 0.5

        # --- History trail: a stack of banded line items (drawn first so the
        # arrow renders on top). Each band is one thin red PlotDataItem whose
        # pen alpha is recomputed in update() from the fade strength. ---
        self._trail_bands = []
        for _ in range(self._TRAIL_BANDS):
            item = pg.PlotDataItem(
                x=[], y=[],
                pen=pg.mkPen(self._TRAIL_COLOR, width=self._TRAIL_WIDTH),
                connect="finite",  # NaN breaks avoid joining disjoint runs
            )
            self.addItem(item)
            self._trail_bands.append(item)

        # --- Arrow shaft: a line from the origin to the tip (Req 11.1). ---
        self._arrow_line = pg.PlotDataItem(
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            pen=pg.mkPen(self._ARROW_COLOR, width=2),
        )
        self.addItem(self._arrow_line)

        # --- Arrow head: positioned at the tip each tick via setPos. ---
        # angle=0 with the default style points the head along +x; update()
        # rotates it to face away from the origin along the vector direction.
        self._arrow_head = pg.ArrowItem(
            angle=0.0,
            headLen=14,
            tipAngle=30,
            pen=pg.mkPen(self._ARROW_COLOR),
            brush=pg.mkBrush(*self._ARROW_COLOR),
        )
        self._arrow_head.setPos(0.0, 0.0)
        self.addItem(self._arrow_head)

    def set_trail_fade(self, fade: float) -> None:
        """Set how quickly the trail fades with age (``0``..``1``).

        ``0`` keeps the whole trail near full brightness (long lingering tail);
        ``1`` makes older portions drop to transparent quickly (short tail).
        The change takes effect on the next :meth:`update`.
        """
        self._trail_fade = max(0.0, min(1.0, fade))

    def _band_alpha(self, band_index: int) -> int:
        """Alpha for band ``band_index`` (0 = oldest, last = newest).

        A fade exponent derived from :attr:`_trail_fade` shapes the ramp: a
        higher fade pushes older bands toward zero alpha more aggressively.
        """
        n = self._TRAIL_BANDS
        # Normalized age position: 0 for the oldest band, 1 for the newest.
        pos = band_index / (n - 1) if n > 1 else 1.0
        # Map fade in [0,1] to an exponent in roughly [0.3 .. ~8]: low fade ->
        # near-flat (tail stays bright), high fade -> steep (tail dies fast).
        exponent = 0.3 + self._trail_fade * 7.0
        return int(self._ALPHA_MAX * (pos ** exponent))

    def update(self, engine):
        """Refresh the arrow and the fading red trail line from the engine.

        Reads the current vector tip ``(engine.N.n1, engine.N.n2)`` and redraws
        the cyan arrow (shaft + oriented head, Req 11.1). Reads
        ``engine.history.trail_points()`` and renders the tip's recent path as a
        thin red polyline split into age bands, each band's alpha set from the
        fade strength so the line fades from bright (near the tip) to
        transparent (oldest), Req 11.2. An empty trail clears the bands.

        Args:
            engine: The :class:`~snn_vector_sim.core.engine.SimulationEngine`
                whose ``N`` and ``history`` are read. No engine state is mutated.
        """
        n1 = engine.N.n1
        n2 = engine.N.n2

        # Arrow shaft from origin to the current tip (Req 11.1).
        self._arrow_line.setData([0.0, n1], [0.0, n2])

        # Position the arrowhead at the tip and orient it along the vector.
        self._arrow_head.setPos(n1, n2)
        if n1 != 0.0 or n2 != 0.0:
            theta = np.degrees(np.arctan2(n2, n1))
            self._arrow_head.setStyle(angle=180.0 - theta)

        # Fading red trail line, split across the age bands (Req 11.2).
        tn1, tn2 = engine.history.trail_points()
        self._draw_trail(tn1, tn2)

    def _draw_trail(self, tn1, tn2) -> None:
        """Render the trail polyline into the banded line items with a fade.

        The trail points (oldest->newest) are partitioned into
        :attr:`_TRAIL_BANDS` contiguous chunks. Each chunk is drawn by one
        PlotDataItem at that band's alpha. Chunks overlap by one point so the
        bands join seamlessly into one continuous line.
        """
        count = len(tn1)
        n = self._TRAIL_BANDS
        if count < 2:
            # Nothing to draw as a line yet; clear every band.
            for item in self._trail_bands:
                item.setData([], [])
            return

        # Split indices 0..count-1 into n contiguous bands.
        edges = np.linspace(0, count - 1, n + 1).astype(int)
        for b, item in enumerate(self._trail_bands):
            lo = edges[b]
            hi = edges[b + 1]
            if hi <= lo:
                item.setData([], [])
                continue
            # Include the next point so adjacent bands connect without a gap.
            xs = tn1[lo:hi + 1]
            ys = tn2[lo:hi + 1]
            r, g, bl = self._TRAIL_COLOR
            pen = pg.mkPen(
                r, g, bl, self._band_alpha(b), width=self._TRAIL_WIDTH,
            )
            item.setData(xs, ys, pen=pen)
