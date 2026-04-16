import numpy as np
from TOOLS.logger import logger


class SyncAlgo:
    def __init__(
        self,
        algo_type: str = "memory",
        cycle_time_s: float = 2.0,
        phase_levels: int = 128,
        K: float = 0.5,
        self_phase_weight: float = 0.25,
        max_memory_cycles: int = 60,
        initial_theta: float | None = None,
    ):
        self.algo_type = algo_type
        self.cycle_time_s = cycle_time_s
        self.phase_levels = phase_levels
        self.K = K

        # Kuramoto / generic sync state
        self.theta = float(
            initial_theta if initial_theta is not None else 2.0 * np.pi * np.random.rand()
        )
        self.kuramoto_conf = 0.5

        # Memory-based sync state
        self.self_phase_weight = self_phase_weight
        self.max_memory_cycles = max_memory_cycles
        self.phase_memory = []

        self.phase_cont = ((self.theta % (2.0 * np.pi)) / (2.0 * np.pi)) * self.phase_levels
        self.prev_phase_cont = self.phase_cont
        self.prev_phase_level = int(round(self.phase_cont)) % self.phase_levels

        self.internal_clock = (self.phase_cont / self.phase_levels) * self.cycle_time_s

    def get_internal_clock(self) -> float:
        return self.internal_clock

    def get_theta(self) -> float:
        return self.theta

    def theta_to_payload(self) -> int:
        return int((self.theta % (2.0 * np.pi)) / (2.0 * np.pi) * self.phase_levels) % self.phase_levels

    def payload_to_theta(self, payload: int) -> float:
        return (payload / float(self.phase_levels)) * (2.0 * np.pi)

    def payload_to_phase_level(self, payload: int) -> int:
        return int(payload) % self.phase_levels

    def phase_level_to_theta(self, phase_level: int) -> float:
        return (float(phase_level % self.phase_levels) / self.phase_levels) * (2.0 * np.pi)

    def phase_level_to_internal_clock(self, phase_level: int) -> float:
        return (float(phase_level % self.phase_levels) / self.phase_levels) * self.cycle_time_s

    def wrap_phase_level(self, phase_level: float) -> int:
        return int(np.floor(phase_level)) % self.phase_levels

    def angle_diff(self, a: float, b: float) -> float:
        return (a - b + np.pi) % (2.0 * np.pi) - np.pi

    def update_kuramoto_confidence(self, theta_j: float, theta: float):
        diff = abs((theta_j - theta + np.pi) % (2.0 * np.pi) - np.pi)

        if diff < 0.1:
            self.kuramoto_conf = min(1.0, self.kuramoto_conf + 0.015)
        elif diff < 0.5:
            self.kuramoto_conf = min(1.0, self.kuramoto_conf + 0.0045)
        elif diff < 1.0:
            self.kuramoto_conf = max(0.0, self.kuramoto_conf + 0.0015)
        else:
            self.kuramoto_conf = max(0.0, self.kuramoto_conf - 0.001)

        self.kuramoto_conf = max(0.0, min(1.0, self.kuramoto_conf))

    def kuramoto_update_basic(self, msgs: list, dt_s: float):
        T = float(self.cycle_time_s)
        omega = (2.0 * np.pi) / T
        theta_dot = omega
        s = 0.0

        for msg in msgs:
            theta_j = self.payload_to_theta(msg.payload)
            s += np.sin(theta_j - self.theta)

        s /= len(msgs) if msgs else 1
        theta_dot += self.K * s

        self.theta = (self.theta + theta_dot * dt_s) % (2.0 * np.pi)
        self.internal_clock = (self.theta / (2.0 * np.pi)) * T
        return self.internal_clock, self.theta

    def kuramoto_update_confidence_based(self, msgs: list, dt_s: float):
        T = float(self.cycle_time_s)
        omega = (2.0 * np.pi) / T
        theta_dot = omega

        for msg in msgs:
            theta_j = self.payload_to_theta(msg.payload)
            self.update_kuramoto_confidence(theta_j, self.theta)
            theta_dot += (self.K * (1 - self.kuramoto_conf)) * np.sin(theta_j - self.theta)

        self.theta = (self.theta + theta_dot * dt_s) % (2.0 * np.pi)
        self.internal_clock = (self.theta / (2.0 * np.pi)) * T
        return self.internal_clock, self.theta

    def kuramoto_update_local_error_based(self, msgs: list, dt_s: float):
        T = float(self.cycle_time_s)
        omega = (2.0 * np.pi) / T
        theta_dot = omega

        if msgs:
            s = 0.0
            err = 0.0
            n = 0

            for msg in msgs:
                theta_j = self.payload_to_theta(msg.payload)
                d = self.angle_diff(theta_j, self.theta)

                s += np.sin(d)
                err += abs(d)
                n += 1

            s /= n
            err /= n

            err_norm = min(1.0, err / np.pi)

            K_min = 0.05
            K_max = self.K
            K_eff = K_min + (K_max - K_min) * err_norm

            theta_dot += K_eff * s

        self.theta = (self.theta + theta_dot * dt_s) % (2.0 * np.pi)
        self.internal_clock = (self.theta / (2.0 * np.pi)) * T
        return self.internal_clock, self.theta

    def advance_phase_level_by_elapsed_time(self, phase_level: int, elapsed_s: float) -> int:
        levels_advanced = int(round((elapsed_s / self.cycle_time_s) * self.phase_levels))
        return (phase_level + levels_advanced) % self.phase_levels

    def linear_memory_weight(self, age_s: float) -> float:
        cycle_age = int(age_s / self.cycle_time_s)
        if cycle_age >= self.max_memory_cycles:
            return 0.0
        if cycle_age <= self.max_memory_cycles / 2:
            weight = 1.0
        else:
            weight = 1.0 - ((cycle_age - self.max_memory_cycles / 2) / (self.max_memory_cycles / 2))
        return max(0.0, weight)

    def cleanup_phase_memory(self, current_time_s: float):
        max_age_s = self.max_memory_cycles * self.cycle_time_s
        self.phase_memory = [
            entry for entry in self.phase_memory
            if (current_time_s - entry["time_s"]) < max_age_s
        ]

    def store_sync_messages(self, synch_msgs: list, current_time_s: float):
        for msg in synch_msgs:
            self.phase_memory.append({
                "phase": self.payload_to_phase_level(msg.payload),
                "time_s": current_time_s,
                "captor_id": msg.captor_id
            })

    def phase_levels_are_similar(self, p1: int, p2: int, tolerance: int = 1) -> bool:
        diff = abs(((p1 - p2 + self.phase_levels // 2) % self.phase_levels) - self.phase_levels // 2)
        return diff <= tolerance

    def weighted_circular_mean_phase_level(self, own_phase_level: int, current_time_s: float) -> int:
        self.cleanup_phase_memory(current_time_s)

        deduped_msgs = []

        for entry in reversed(self.phase_memory):
            age_s = current_time_s - entry["time_s"]
            w = self.linear_memory_weight(age_s)
            if w <= 0.0:
                continue

            projected_phase = self.advance_phase_level_by_elapsed_time(entry["phase"], age_s)
            captor_id = entry["captor_id"]

            redundant = False
            for kept in deduped_msgs:
                if kept["captor_id"] == captor_id and self.phase_levels_are_similar(
                    projected_phase, kept["projected_phase"], tolerance=1
                ):
                    redundant = True
                    break

            if not redundant:
                deduped_msgs.append({
                    "captor_id": captor_id,
                    "projected_phase": projected_phase,
                    "weight": w
                })

        if not deduped_msgs:
            return own_phase_level

        ext_weights = np.array([msg["weight"] for msg in deduped_msgs], dtype=float)
        ext_weights_sum = np.sum(ext_weights)

        if ext_weights_sum <= 0.0:
            return own_phase_level

        ext_weights = ((1.0 - self.self_phase_weight) / ext_weights_sum) * ext_weights

        own_angle = (own_phase_level / self.phase_levels) * (2.0 * np.pi)
        z = self.self_phase_weight * np.exp(1j * own_angle)

        for msg, w in zip(deduped_msgs, ext_weights):
            angle = (msg["projected_phase"] / self.phase_levels) * (2.0 * np.pi)
            z += w * np.exp(1j * angle)

        if np.abs(z) < 1e-12:
            return own_phase_level

        mean_angle = np.angle(z)
        if mean_angle < 0.0:
            mean_angle += 2.0 * np.pi

        new_phase_level = int(round((mean_angle / (2.0 * np.pi)) * self.phase_levels)) % self.phase_levels

        logger.log(
            "DEBUG",
            f"Weighted mean: own={own_phase_level}, kept_msgs={len(deduped_msgs)}, "
            f"raw_memory={len(self.phase_memory)}, new={new_phase_level}"
        )

        return new_phase_level

    def update_memory_based_sync(self, synch_msgs: list, time_s: float, dt_s: float):
        self.store_sync_messages(synch_msgs, time_s)

        self.prev_phase_cont = self.phase_cont
        phase_increment = (dt_s / self.cycle_time_s) * self.phase_levels
        self.phase_cont = (self.phase_cont + phase_increment) % self.phase_levels

        wrapped = self.phase_cont < self.prev_phase_cont

        if wrapped:
            current_phase_level = int(round(self.phase_cont)) % self.phase_levels
            new_phase_level = self.weighted_circular_mean_phase_level(current_phase_level, time_s)

            logger.log(
                "DEBUG",
                f"Memory sync wrap: old_phase={current_phase_level}, new_phase={new_phase_level}, "
                f"memory_size={len(self.phase_memory)}"
            )
            self.phase_cont = float(new_phase_level)

        current_phase_level = int(round(self.phase_cont)) % self.phase_levels
        self.prev_phase_level = current_phase_level
        self.theta = self.phase_level_to_theta(current_phase_level)
        self.internal_clock = (self.phase_cont / self.phase_levels) * self.cycle_time_s

        return self.internal_clock, self.theta


    def update(self, synch_msgs: list, time_s: float, dt_s: float):
        if self.algo_type == "memory":
            return self.update_memory_based_sync(synch_msgs, time_s, dt_s)

        if self.algo_type == "kuramoto_basic":
            return self.kuramoto_update_basic(synch_msgs, dt_s)

        if self.algo_type == "kuramoto_confidence":
            return self.kuramoto_update_confidence_based(synch_msgs, dt_s)

        if self.algo_type == "kuramoto_local_error":
            return self.kuramoto_update_local_error_based(synch_msgs, dt_s)

        raise ValueError(f"Unknown sync algorithm: {self.algo_type}")