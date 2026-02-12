from PyQt6.QtCore import Qt, QTimer
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from GRAPHICS.viewer import Viewer
from EXP.experiment import Exp

class Engine( QWidget ):
    def __init__(self, viewer: Viewer, delta_t_ms):
        super().__init__()
        self.viewer            = viewer
        self.isAllTrialsInit   = False
        self.isSingleTrialInit = False
            
        self.time_ms    = 0
        self.delay_ms   = 0
        self.delta_t_ms = delta_t_ms + self.delay_ms # in ms

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
    
    def set_delay_on_delta_t_ms(self, value):
        self.delay_ms = value
        
    def start_timer(self):
        if not self.timer.isActive():
            self.timer.start(self.delta_t_ms + self.delay_ms)
            
    def stop_main_timer(self):
        self.timer.stop()
        self.stop_timer.stop()

    def tick(self):
        self.time_ms += self.delta_t_ms
        # self.iterations = int(self.time_ms/self.delta_t_ms)
        self.advance()
        self.label.setText(f"Time: {self.time_ms} ms, Trial = {Exp.trial}, Iteration = {Exp.iter}")
        self.viewer.update()
        
        
    def step_by_step_interval_ms(self):
        # Star main timer
        self.timer.start(self.delta_t_ms + self.delay_ms)
        # Afer interval ms → automatic stop
        self.stop_timer.start(self.delta_t_ms + self.delay_ms)
        
                
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