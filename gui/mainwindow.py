import os

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QLabel, QWidget
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QTimer

from .tabs.simulation      import SimulationTab
from .tabs.crystals        import CrystalsTab
from .tabs.results         import ResultsTab
from .tabs.sweep           import SweepTab
from .tabs.phase_matching  import PhaseMatchingTab
from .tabs.full_simulation import FullSimulationTab
from .tabs.full_results    import FullResultsTab
from .core_py.config_builder import build_config, config_summary
from .core_py.engine_runner  import EngineRunner


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BRAHMS — Three-Wave Mixing Simulator")
        self.setMinimumSize(1100, 720)
        self.resize(1400, 860)

        _icon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "icon", "icon.png")
        if os.path.isfile(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))

        self._build_tabs()
        self._build_status_bar()
        self._check_gpu()

    # ── Tabs ────────────────────────────────────────────────────────────
    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        self.crystal_tab      = CrystalsTab()
        self.pm_tab           = PhaseMatchingTab()
        self.sim_tab          = SimulationTab()
        self.results_tab      = ResultsTab()
        self.full_sim_tab     = FullSimulationTab()
        self.full_results_tab = FullResultsTab()
        self.sweep_tab        = SweepTab(sim_tab_ref=self.sim_tab,
                                          crystal_tab_ref=self.crystal_tab)

        # Single simulation: inner tab widget with "Set Parameters" and "Results"
        self.single_sim_tabs = QTabWidget()
        self.single_sim_tabs.setDocumentMode(True)
        self.single_sim_tabs.addTab(self.sim_tab,     "  Set Parameters  ")
        self.single_sim_tabs.addTab(self.results_tab, "  Results  ")

        # (3+1)D Simulations: inner tab widget with "Set Parameters" and "Results"
        self.full_3d_tabs = QTabWidget()
        self.full_3d_tabs.setDocumentMode(True)
        self.full_3d_tabs.addTab(self.full_sim_tab,     "  Set Parameters  ")
        self.full_3d_tabs.addTab(self.full_results_tab, "  Results  ")

        self.tabs.addTab(self.crystal_tab,     "  Nonlinear Crystals  ")
        self.tabs.addTab(self.pm_tab,          "  Phase Matching  ")
        self.tabs.addTab(self.single_sim_tabs, "  Single simulation  ")
        self.tabs.addTab(self.sweep_tab,       "  Parameter Sweep  ")
        self.tabs.addTab(self.full_3d_tabs,    "  (3+1)D Simulations  ")

        self.setCentralWidget(self.tabs)

        # Which tab triggered the running simulation ("sim" or "full")
        self._active_run_source = "sim"

        # Forward run signals from simulation tab
        self.sim_tab.run_requested.connect(self._on_run_requested)
        self.sim_tab.stop_requested.connect(self._stop)

        # Forward run signals from full simulation tab
        self.full_sim_tab.run_requested.connect(self._on_full_run_requested)
        self.full_sim_tab.stop_requested.connect(self._stop)

        # Phase-matching → Simulation Parameters (and Full Simulation)
        self.pm_tab.pm_wavelengths_found.connect(self._on_pm_wavelengths_found)

        # Engine runner (persistent — keeps track of compiled grid)
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._runner = EngineRunner(_project_root, parent=self)
        self._runner.log_line.connect(self.sim_tab.append_log)
        self._runner.log_line.connect(self.full_sim_tab.append_log)
        self._runner.finished.connect(self._on_simulation_finished)

    # ── Status bar ──────────────────────────────────────────────────────
    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self.lbl_status = QLabel("Ready")
        self.lbl_gpu    = QLabel("GPU: detecting…")
        self.lbl_gpu.setStyleSheet("color: #777777;")

        sb.addWidget(self.lbl_status)
        sb.addPermanentWidget(self.lbl_gpu)

    def _check_gpu(self):
        """Non-blocking GPU check."""
        def _do_check():
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    name = result.stdout.strip().split("\n")[0]
                    self.lbl_gpu.setText(f"GPU: {name}")
                    self.lbl_gpu.setStyleSheet("color: #00e676;")
                else:
                    self.lbl_gpu.setText("GPU: not found — MPI mode available")
                    self.lbl_gpu.setStyleSheet("color: #ff6d00;")
            except Exception:
                self.lbl_gpu.setText("GPU: —")
        QTimer.singleShot(500, _do_check)

    # ── Slots ───────────────────────────────────────────────────────────
    def _on_run_requested(self, params: dict):
        self._active_run_source = "sim"
        backend = params.get("backend", "cuda")
        self.lbl_status.setText(f"Running [{backend.upper()}]…")
        self.sim_tab.btn_run.setEnabled(False)
        self.sim_tab.btn_mpi.setEnabled(False)
        self.sim_tab.btn_stop.setEnabled(True)

        # Merge absorption and TPA on/off flags from Crystals tab
        params.update(self.crystal_tab.alpha_flags())
        params.update(self.crystal_tab.beta_flags())
        params.update(self.crystal_tab.rho_flags())

        # Build config
        try:
            cfg = build_config(params)
        except Exception as exc:
            self.sim_tab.append_log(f"[ERROR] Config build failed: {exc}")
            self._stop()
            return

        self.sim_tab.append_log(config_summary(cfg))

        # Output directory: <project_root>/output/
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(_root, "output")

        grid = params.get("grid", {"NX": 128, "NY": 128, "NZ": 100, "NT": 512})
        self._runner.run(cfg, grid, out_dir, backend=backend,
                          omp_threads=params.get("omp_threads"))

    def _on_full_run_requested(self, params: dict):
        self._active_run_source = "full"
        backend = params.get("backend", "cuda")
        self.lbl_status.setText(f"Running [{backend.upper()}] (3+1)D…")
        self.full_sim_tab.btn_run.setEnabled(False)
        self.full_sim_tab.btn_mpi.setEnabled(False)
        self.full_sim_tab.btn_stop.setEnabled(True)

        params.update(self.crystal_tab.alpha_flags())
        params.update(self.crystal_tab.beta_flags())
        params.update(self.crystal_tab.rho_flags())

        try:
            cfg = build_config(params)
        except Exception as exc:
            self.full_sim_tab.append_log(f"[ERROR] Config build failed: {exc}")
            self._stop()
            return

        self.full_sim_tab.append_log(config_summary(cfg))

        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(_root, "output_full")

        grid = params.get("grid", {"NX": 128, "NY": 128, "NZ": 200, "NT": 256})
        self._runner.run(cfg, grid, out_dir, backend=backend,
                          omp_threads=params.get("omp_threads"))

    def _run_gpu(self):
        self.tabs.setCurrentWidget(self.single_sim_tabs)
        self.single_sim_tabs.setCurrentWidget(self.sim_tab)
        self.sim_tab._on_run_gpu()

    def _run_mpi(self):
        self.tabs.setCurrentWidget(self.single_sim_tabs)
        self.single_sim_tabs.setCurrentWidget(self.sim_tab)
        self.sim_tab._on_run_mpi()

    def _on_simulation_finished(self, success: bool, out_dir: str):
        # Re-enable whichever tab triggered the run
        if self._active_run_source == "full":
            self.full_sim_tab.btn_run.setEnabled(True)
            self.full_sim_tab.btn_mpi.setEnabled(True)
            self.full_sim_tab.btn_stop.setEnabled(False)
            if success:
                self.lbl_status.setText("Done.")
                self.full_results_tab.load_output_dir(out_dir)
                self.tabs.setCurrentWidget(self.full_3d_tabs)
                self.full_3d_tabs.setCurrentWidget(self.full_results_tab)
            else:
                self.lbl_status.setText("Failed — see log.")
        else:
            self.sim_tab.btn_run.setEnabled(True)
            self.sim_tab.btn_mpi.setEnabled(True)
            self.sim_tab.btn_stop.setEnabled(False)
            if success:
                self.lbl_status.setText("Done.")
                self.results_tab.load_output_dir(out_dir)
                self.tabs.setCurrentWidget(self.single_sim_tabs)
                self.single_sim_tabs.setCurrentWidget(self.results_tab)
            else:
                self.lbl_status.setText("Failed — see log.")

    def _on_pm_wavelengths_found(self, lp: float, ls: float, li: float,
                                  T: float, Lambda: float):
        self.sim_tab.load_pm_wavelengths(lp, ls, li, T, Lambda)
        self.tabs.setCurrentWidget(self.single_sim_tabs)
        self.single_sim_tabs.setCurrentWidget(self.sim_tab)
        self.lbl_status.setText(
            f"PM loaded: λ_p={lp:.4f} μm  λ_s={ls:.4f} μm  λ_i={li:.4f} μm  T={T:.1f} °C")

    def _stop(self):
        self._runner.stop()
        self.lbl_status.setText("Stopped.")
        for tab in (self.sim_tab, self.full_sim_tab):
            tab.btn_run.setEnabled(True)
            tab.btn_mpi.setEnabled(True)
            tab.btn_stop.setEnabled(False)

