from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque
from TOOLS.angle_to_sector import angle_to_sector
import numpy as np



@dataclass(frozen=True)
class IRMessage:
    sender_id: int
    payload: int          # 0..255 (8 bits)
    time_s: float
    captor_id: int
    strenght: float = 1.0     # optionnel (0..1)
    

@dataclass
class IRCommConfig:
    range_m: float = 0.5
    robot_rad_m: float = 0.15
    fov_deg: float = 180.0
    num_captors: int = 6
    max_process_rate_s: float = 6.0
    msg_ttl_s: float = 0.5
    max_inbox: int = 3 
    drop_prob: float = 0.0 
    enabled: bool = True


class IRComm:
    """
    Module de communication attaché à 1 robot.
    - inbox: file de messages reçus 
    - consume(): retourne une liste de messages (limité à max_process_rate_s)
    - send(): envoie un message dans le medium
    """
    def __init__(self, robot_id: int, config: IRCommConfig):
        self.robot_id = int(robot_id)
        self.cfg = config

        self._inbox: Deque[IRMessage] = deque()
        self._tokens: float = 0.0
        self._last_time_s: Optional[float] = None

        self._outbox: List[Tuple[int, int, float]] = []  # (sender_id, payload, time_s)

    def reset(self):
        self._inbox.clear()
        self._outbox.clear()
        self._tokens = 0.0
        self._last_time_s = None

    def send(self, payload: int, time_s: float):
        if not self.cfg.enabled:
            return
        payload = int(payload) & 0xFF  
        self._outbox.append((self.robot_id, payload, float(time_s)))

    def _pop_outbox(self) -> List[Tuple[int, int, float]]:
        out = self._outbox
        self._outbox = []
        return out

    def _push_inbox(self, msg: IRMessage):
        if not self.cfg.enabled:
            return
        if len(self._inbox) >= int(self.cfg.max_inbox):
            # drop si plein
            return
        self._inbox.append(msg)

    def consume(self, time_s: float, dt_s: float) -> List[IRMessage]:
        """
        Retourne au maximum max_process_rate_s messages par seconde (token bucket).
        """
        if not self.cfg.enabled:
            return []

        time_s = float(time_s)
        dt_s = float(dt_s)

        # token bucket
        if self._last_time_s is None:
            self._last_time_s = time_s

        # On crédite en tokens selon dt
        self._tokens += self.cfg.max_process_rate_s * dt_s

        #accumulation de token (max 0.5s)
        self._tokens = min(self._tokens, self.cfg.max_process_rate_s * 0.5)

        # purge messages expirés
        ttl = float(self.cfg.msg_ttl_s)
        while self._inbox and (time_s - self._inbox[0].time_s) > ttl:
            self._inbox.popleft()

        n = int(min(len(self._inbox), np.floor(self._tokens)))
        msgs = []
        for _ in range(n):
            msgs.append(self._inbox.popleft())
        self._tokens -= n
        return msgs


class IRMedium:
    """
    Médium IR global: distribue les messages envoyés par les robots.
    pram:
    - Portée fixe (range_m)
    - FOV demi-cercle devant le robot récepteur (fov_deg)
    """
    def __init__(self, config: IRCommConfig):
        self.cfg = config

    @staticmethod
    def _angle_wrap_pi(a: float) -> float:
        return (a + np.pi) % (2.0 * np.pi) - np.pi


    def step(self, robots: np.ndarray, time_s: float, dt_s: float):

        if not self.cfg.enabled:
            return

        time_s = float(time_s)

        # 1) Collecter tous les messages envoyés cette frame
        emitted: List[Tuple[int, int, float]] = []
        for rb in robots:
            if not hasattr(rb, "ir_comm") or rb.ir_comm is None:
                continue
            emitted.extend(rb.ir_comm._pop_outbox())

        if not emitted:
            return

        # 2) Distribution: pour chaque msg, tester tous les receveurs
        fov_rad_half = np.deg2rad(self.cfg.fov_deg) * 0.5

        range_m = float(self.cfg.range_m )

        # Précompute positions/yaw pour efficacité
        pos = {int(rb.id): np.array(rb.pos[0:2], dtype=float) for rb in robots}
        yaw = {int(rb.id): float(rb.rot[2]) for rb in robots}

        for sender_id, payload, t_send in emitted:
            sender_id = int(sender_id)
            if sender_id not in pos:
                continue

            Ps = pos[sender_id]

            for rb in robots:
                rid = int(rb.id)
                if rid == sender_id:
                    continue
                if not hasattr(rb, "ir_comm") or rb.ir_comm is None:
                    continue

                Pr = pos[rid]
                v = Ps - Pr
                d_c = float(np.linalg.norm(v))  # distance depuis le centre des robots
                d = max(0.0, d_c - 2*float(self.cfg.robot_rad_m))  # distance depuis les bords des robots
                if d > range_m:
                    continue

                # Angle entre heading du receveur et direction vers l'émetteur
                ang_to_sender = float(np.arctan2(v[1], v[0]))
                rel_r = self._angle_wrap_pi(ang_to_sender - yaw[rid])

                # hors du demi-cercle devant le receveur 
                if abs(rel_r) > fov_rad_half:
                    continue

                # Angle entre heading de l'émetteur et direction vers le receveur
                ang_to_reciever = float(np.arctan2(-v[1], -v[0]))
                rel_s = self._angle_wrap_pi(ang_to_reciever - yaw[sender_id])

                #hors du demi-cercle devant l'émetteur
                if abs(rel_s) > fov_rad_half:
                    continue

                # Option bruit: drop
                if self.cfg.drop_prob > 0.0 and np.random.rand() < self.cfg.drop_prob:
                    continue
                
                # déterminer secteur (0..5)
                fov_rad = 2.0 * fov_rad_half
                sector = angle_to_sector(rel_r, fov_rad, self.cfg.num_captors)
                # strenght simple (optionnel): 1 à 0 selon distance
                strenght = max(0.0, 1.0 - d / range_m)

                rb.ir_comm._push_inbox(IRMessage(sender_id=sender_id,
                                                payload=int(payload), 
                                                time_s=float(t_send),
                                                captor_id=sector,
                                                strenght=strenght))
