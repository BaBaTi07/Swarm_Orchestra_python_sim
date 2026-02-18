from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QToolButton, QVBoxLayout, QHBoxLayout, QSpinBox, QLabel, QFileDialog

from GRAPHICS.viewer import Viewer
from GRAPHICS.engine import Engine
from EXP.experiment import Exp


class MainWindow(QMainWindow):
    delta_t_ms = 0.0
    
    def __init__(self, delta_t_ms):
        # Configure main window 
        super().__init__()
        self.viewer = Viewer( )
        self.eng    = Engine( self.viewer, delta_t_ms)
        # self.viewer.setMinimumHeight(300);
        self.viewer.resize(600, 600)
        self.setWindowTitle("E-puck simulator (version 0.1)")
        self.setMinimumSize(QSize(1000, 800))
        
        # Create Buttons
        self.buttonslayout = QHBoxLayout()
        self.create_buttons()
        
        # Create the window layout
        self.mainlayout = QVBoxLayout()
        # self.mainlayout.setContentsMargins(0,0,0,0)
        # self.mainlayout.setSpacing(0)
        
        self.mainlayout.addWidget(self.viewer, 1) 
        self.mainlayout.addLayout(self.buttonslayout )
        
        # Set the central widget of the Window.
        container = QWidget()
        container.setLayout(self.mainlayout)
        self.setCentralWidget(container)
    
    def reset_run_buttons( self ):    
        self.button_2.setText("Run")
        self.button_2.setChecked(False)

    def create_buttons( self ):
        self.button_1 = QToolButton()
        #  self.button_1.setCheckable(True)
        self.button_1.setText("Step-by-Step")
        # oneStepButtom.setIcon(QIcon('path/to/icon.png'))
        self.button_1.clicked.connect( self.advance_step_by_step )
        
        self.button_2 = QToolButton()
        self.button_2.setCheckable(True)
        self.button_2.setText("Run")
        # oneStepButtom.setIcon(QIcon('path/to/icon.png'))
        self.button_2.clicked.connect( self.advance_run )
        
        self.button_3 = QToolButton()
        self.button_3.setText("Init")
        # oneStepButtom.setIcon(QIcon('path/to/icon.png'))
        self.button_3.clicked.connect(self.re_init)
        
        self.button_4 = QToolButton()
        self.button_4.setText("Quit")
        # oneStepButtom.setIcon(QIcon('path/to/icon.png'))
        self.button_4.clicked.connect(self.close)

        self.button_fast = QToolButton()
        self.button_fast.setText("Speed")
        self.button_fast.clicked.connect(self.toggle_fast_mode)

        # Load JSON button
        self.button_load = QToolButton()
        self.button_load.setText("Load JSON")
        self.button_load.clicked.connect(self.load_json)

        
        # # Input time interval
        self.spin_ms = QSpinBox()
        self.spin_ms.setRange(0, 90)
        self.spin_ms.setValue(0)
        self.spin_ms.setSuffix(" ms slower")
        self.spin_ms.valueChanged.connect(self.on_value_changed)
        self.delay_label = QLabel(f"Actual delay (ms): {self.spin_ms.value()}")
        
        # self.spin_ms = QSpinBox()
        # # self.spin_ms.setRange(self.delta_t_ms)
        # self.spin_ms.setSingleStep(0)
        # self.spin_ms.setValue(self.delta_t_ms)
        # self.spin_ms.setSuffix(" times speeded up")
        
        # QWidget added to buttom layout
        self.buttonslayout.addWidget(self.button_1)
        self.buttonslayout.addWidget(self.button_2)
        self.buttonslayout.addWidget(self.button_3)
        self.buttonslayout.addWidget(self.button_4)
        self.buttonslayout.addWidget(self.button_fast)
        self.buttonslayout.addWidget(self.delay_label)
        self.buttonslayout.addWidget(self.spin_ms)
        self.buttonslayout.addWidget(self.eng.label)
        self.buttonslayout.addWidget(self.button_load)
        
    def re_init( self ):
        self.eng.re_init()
        
    def advance_step_by_step( self ):
        self.eng.step_by_step_interval_ms()

    def toggle_fast_mode( self ):
        if self.eng.speed_multiplier == 1.0:
            self.eng.set_speed_multiplier(4.0)
            self.button_fast.setText("Fast")
        elif self.eng.speed_multiplier < 1.0:
            self.eng.set_speed_multiplier(1.0)
            self.button_fast.setText("Base")
        elif self.eng.speed_multiplier > 1.0:
            self.eng.set_speed_multiplier(0.25)
            self.button_fast.setText("Slow")
            
    def advance_run( self, checked ):
        if checked:
            self.button_1.setEnabled(False)
            self.button_2.setText("Stop trial")
            self.button_3.setEnabled(False)
            self.button_4.setEnabled(False)
            self.spin_ms.setEnabled(False)
            self.eng.start_timer()
        else:
            self.button_1.setEnabled(True)
            self.button_2.setText("Run")
            self.button_3.setEnabled(True)
            self.button_4.setEnabled(True)
            self.spin_ms.setEnabled(True)
            self.eng.stop_main_timer()
        
    def on_value_changed(self, value):
        self.delay_label.setText(f"Actual delay (ms): {value}")
        self.eng.set_delay_on_delta_t_ms(value)

    def load_json(self):
        # Open a file dialog to select a JSON file from /json_files directory
        path, _ = QFileDialog.getOpenFileName(self, "Load JSON File", "./json_files", "JSON Files (*.json)")

        if path:
            self.eng.reload_experiment(path)
            self.reset_run_buttons()


    def close( self ):
        QApplication.instance().quit()
    
if __name__ == "__main__" :
    # create the QApplication
    app = QApplication([])

    # create the main window
    window = MainWindow()
    window.show()

    # start the event loop
    app.exec()