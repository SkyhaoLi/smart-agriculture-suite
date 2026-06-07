"""SimClock — simulated time with configurable acceleration."""

import time
import threading


class SimClock:
    """
    Provides millis() that returns simulated milliseconds.
    Time acceleration factor allows 1x (real-time) up to 3600x (1 hour/sec).
    """

    def __init__(self, time_scale: float = 1.0):
        self._lock = threading.Lock()
        self._time_scale = time_scale
        self._real_start = time.monotonic()
        self._sim_offset = 0.0  # accumulated simulated ms before last scale change
        self._scale_change_real = self._real_start

    def millis(self) -> int:
        """Return current simulated milliseconds (matches ESP32 millis())."""
        with self._lock:
            now = time.monotonic()
            elapsed_real = now - self._scale_change_real
            elapsed_sim = elapsed_real * 1000.0 * self._time_scale
            return int(self._sim_offset + elapsed_sim)

    def set_time_scale(self, scale: float):
        """Change the time acceleration factor, preserving accumulated time."""
        with self._lock:
            now = time.monotonic()
            elapsed_real = now - self._scale_change_real
            self._sim_offset += elapsed_real * 1000.0 * self._time_scale
            self._scale_change_real = now
            self._time_scale = scale

    @property
    def time_scale(self) -> float:
        with self._lock:
            return self._time_scale

    def sim_hours(self) -> int:
        """Current simulated hour of day (0-23), assuming day 0 starts at 0:00."""
        return (self.millis() // 3600000) % 24

    def sim_elapsed_hours(self) -> float:
        """Total elapsed simulated hours since clock was created."""
        return self.millis() / 3600000.0

    def is_day(self) -> bool:
        """True if simulated time is 6:00-18:00."""
        h = self.sim_hours()
        return 6 <= h < 18

    def uptime_seconds(self) -> float:
        """Real wall-clock seconds since clock was created."""
        return time.monotonic() - self._real_start

    def tick(self, advance_ms: int):
        """Advance simulated clock by advance_ms milliseconds (for batch simulation)."""
        with self._lock:
            now = time.monotonic()
            elapsed_real = now - self._scale_change_real
            self._sim_offset += elapsed_real * 1000.0 * self._time_scale
            self._sim_offset += advance_ms
            self._scale_change_real = now
