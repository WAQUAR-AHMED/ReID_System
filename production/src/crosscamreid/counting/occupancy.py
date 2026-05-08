"""
occupancy.py
============
Threshold + edge-triggered alerting on top of `current_occupancy`.

Status levels (derived from threshold + alert_rate):
  alert_threshold = threshold * alert_rate / 100
  - "normal"        : occupancy < alert_threshold
  - "approaching"   : alert_threshold <= occupancy < threshold
  - "at_capacity"   : occupancy == threshold
  - "over_capacity" : occupancy > threshold

Alerts fire on the rising edge only:
  - "approaching"  : crossed alert_threshold from below
  - "at_capacity"  : crossed threshold from below
  - "over_capacity": went past threshold

Falling back below a level does not re-trigger; we only re-fire on a fresh
rising edge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OccupancySnapshot:
    percentage: float
    threshold: int
    over_capacity: int
    status: str
    alert_triggered: bool
    alert_type: str | None


class OccupancyTracker:
    def __init__(self, threshold: int, alert_rate: float) -> None:
        self.threshold = max(0, int(threshold))
        self.alert_rate = max(0.0, min(100.0, float(alert_rate)))
        self.alert_threshold_value = self.threshold * (self.alert_rate / 100.0)
        # Highest level we have most-recently *fired* an alert at, so we don't
        # re-fire while the occupancy stays at or above that level. Reset to a
        # lower bucket once we drop below it.
        self._last_fired_level: int = 0   # 0=normal, 1=approaching, 2=at, 3=over

    def update(self, current_occupancy: int) -> OccupancySnapshot:
        n = max(0, int(current_occupancy))
        threshold = self.threshold

        if threshold <= 0:
            return OccupancySnapshot(
                percentage=0.0,
                threshold=threshold,
                over_capacity=0,
                status="normal",
                alert_triggered=False,
                alert_type=None,
            )

        percentage = round((n / threshold) * 100.0, 1)

        if n > threshold:
            level = 3
            status = "over_capacity"
        elif n == threshold:
            level = 2
            status = "at_capacity"
        elif n >= self.alert_threshold_value:
            level = 1
            status = "approaching"
        else:
            level = 0
            status = "normal"

        triggered = level > self._last_fired_level
        # Decay the fired level if we drop down -- otherwise we'd never fire
        # again on a new rising edge.
        if level < self._last_fired_level:
            self._last_fired_level = level
        if triggered:
            self._last_fired_level = level

        alert_type: str | None
        if not triggered:
            alert_type = None
        elif level == 1:
            alert_type = "approaching_threshold"
        elif level == 2:
            alert_type = "at_threshold"
        else:
            alert_type = "over_capacity"

        over = max(0, n - threshold)

        return OccupancySnapshot(
            percentage=percentage,
            threshold=threshold,
            over_capacity=over,
            status=status,
            alert_triggered=bool(triggered),
            alert_type=alert_type,
        )
