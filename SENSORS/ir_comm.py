import numpy as np
from numpy.typing import NDArray

class IRComm:
    """
    Placeholder: communication IR (13 TX, 6 RX).
    Plus tard: portée, FOV demi-cercle avant, bruit, occlusions, etc.
    """
    tx_count = 13
    rx_count = 6
    range_m  = 0.5      # 50 cm
    fov_rad  = np.pi    # demi-cercle avant

    def __init__(self):
        self.rx_buffer: list[dict] = []  # messages reçus

    def clear(self):
        self.rx_buffer.clear()

    def broadcast(self, sender_id: int, payload: dict):
        # plus tard : Arena / world gère qui reçoit
        pass

    def receive(self) -> list[dict]:
        return list(self.rx_buffer)
