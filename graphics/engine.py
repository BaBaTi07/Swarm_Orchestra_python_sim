from PyQt6.QtCore import Qt, QTimer
import numpy as np
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from GRAPHICS.viewer import Viewer
from EXP.experiment import Exp
from WORLD.arena import Arena
from TOOLS.read_json import read_json_file
from TOOLS.logger import logger

class Engine( QWidget ):
    def __init__(self, viewer: Viewer, delta_t_ms):
        super().__init__()
        self.viewer            = viewer
        self.isAllTrialsInit   = False
        self.isSingleTrialInit = False
            
        self.time_ms    = 0
        self.delay_ms   = 0
        self.base_delta_t_ms = float(delta_t_ms)
        self.speed_multiplier = 1.0

        # main Timer (tick every "interval" ms)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        
        # Timer for automatic stop
        self.stop_timer = QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self.stop_main_timer)

        # Label
        self.label = QLabel(f"Time: {self.time_ms} ms, Trial = {Exp.trial}, Iteration = {Exp.iter}")        
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._apply_timer_interval()
    
    def reload_experiment(self, json_path: str):
        
        self.stop_main_timer()

        Arena.reset()
        Exp.reset()

        exp_seed, delta_t_ms = read_json_file(json_path)
        logger.log("DEBUG", f"RELOAD delta_t_ms: {delta_t_ms}, base_delta_t_ms: {self.base_delta_t_ms}")

        if exp_seed is not None:
            np.random.seed(int(exp_seed))
        if delta_t_ms is not None:
            self.base_delta_t_ms = float(delta_t_ms)
            self._apply_timer_interval()

        # 4) Re-init engine state
        self.time_ms = 0
        self.isAllTrialsInit = False
        self.isSingleTrialInit = False
        self.label.setText(f"Time: {self.time_ms} ms, Trial = {Exp.trial}, Iteration = {Exp.iter}")

        # 5) Trigger redraw
        self.viewer.update()

    #internal method to compute the effective interval based on base delta_t, delay and speed multiplier
    def _effective_interval_ms(self) -> int:
        eff = (self.base_delta_t_ms + float(self.delay_ms)) / float(self.speed_multiplier)
        if eff < 1:
            eff = 1
        return int(round(eff))

    def _apply_timer_interval(self) -> None:
        interval = self._effective_interval_ms()
        self.timer.setInterval(interval)
        self.stop_timer.setInterval(interval)

    def set_delay_on_delta_t_ms(self, value):
        self.delay_ms = int(value)
        self._apply_timer_interval()

    def set_speed_multiplier(self, mult):
        mult = float(mult)
        self.speed_multiplier = mult
        self._apply_timer_interval()
        print("speed:", self.speed_multiplier, "timer interval:", self.timer.interval())
        
    def start_timer(self):
        if not self.timer.isActive():
            self.timer.start()
            
    def stop_main_timer(self):
        self.timer.stop()
        self.stop_timer.stop()

    def tick(self):
        self.time_ms += int(round(self.base_delta_t_ms))

        # self.iterations = int(self.time_ms/self.delta_t_ms)
        self.advance()
        self.label.setText(f"Time: {self.time_ms} ms, Trial = {Exp.trial}, Iteration = {Exp.iter}")
        self.viewer.update()
        
        
    def step_by_step_interval_ms(self):
        # Star main timer
        self.timer.start()
        # Afer interval ms → automatic stop
        self.stop_timer.start()
        
                
    def initialize( self ) :
        if self.isAllTrialsInit == False:
            Exp.init_all_trials()
            Exp.init_single_trial()
            self.isAllTrialsInit   = True
            self.isSingleTrialInit = True
        elif self.isSingleTrialInit == False:
            Exp.init_single_trial()
            self.isSingleTrialInit = True  
            
            
    def finalise( self ):
        if not Exp.finalise_single_trial():
            self.isSingleTrialInit = False
        if not Exp.finalise_all_trials():
            self.isAllTrialsInit = False
            self.isSingleTrialInit = False
            self.stop_main_timer()
    
    def re_init(self):
        Exp.trial = Exp.num_trials
        Exp.iter  = Exp.num_iterations
        self.finalise( )
        
    def advance( self ):
        self.initialize()
        Exp.make_iteration()
        self.finalise()    
        
class Tmp_window(QWidget):
    def __init__(self):
        super().__init__()
        self.eng = Engine(  )
        
        self.bottone_start = QPushButton("Start Timer")
        self.bottone_stop = QPushButton("Stop Timer")

        # self.bottone_start.clicked.connect(self.eng.play)
        # self.bottone_stop.clicked.connect(self.eng.stop)

        # layout = QVBoxLayout()
        # layout.addWidget(self.eng.label)
        # layout.addWidget(self.bottone_start)
        # layout.addWidget(self.bottone_stop)
        # self.setLayout(layout)
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tmpwin = Tmp_window()
    tmpwin.show()
    sys.exit(app.exec())