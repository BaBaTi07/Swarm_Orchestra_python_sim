from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QToolButton, QVBoxLayout, QHBoxLayout, QSpinBox

from graphics.viewer import Viewer
from graphics.engine import Engine
from exp.experiment_1 import Exp


class MainWindow(QMainWindow):
    def __init__(self, my_exp: Exp, delta_t_ms ):
        # Configure main window 
        super().__init__()
        self.delta_t_ms = delta_t_ms
        self.viewer = Viewer( my_exp )
        self.eng    = Engine( self.viewer, my_exp, self.delta_t_ms)
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
        self.button_3.setText("Quit")
        # oneStepButtom.setIcon(QIcon('path/to/icon.png'))
        self.button_3.clicked.connect(self.close)
        
        # # Input time interval
        # self.spin_ms = QSpinBox()
        # # self.spin_ms.setRange(self.delta_t_ms)
        # self.spin_ms.setSingleStep(0)
        # self.spin_ms.setValue(self.delta_t_ms)
        # self.spin_ms.setSuffix(" times speeded up")
        
        # QWidget added to buttom layout
        self.buttonslayout.addWidget(self.button_1)
        self.buttonslayout.addWidget(self.button_2)
        self.buttonslayout.addWidget(self.button_3)
        self.buttonslayout.addWidget(self.eng.label)

    def advance_step_by_step( self, checked ):
        self.eng.step_by_step_interval_ms()
            
    def advance_run( self, checked ):
        if checked:
            self.button_1.setEnabled(False)
            self.button_2.setText("Stop trial")
            self.button_3.setEnabled(False)
            self.eng.start_timer()
        else:
            self.button_1.setEnabled(True)
            self.button_2.setText("Run")
            self.button_3.setEnabled(True)
            self.eng.stop_main_timer()
            
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