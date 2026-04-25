from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelProfile:
    name: str
    accuracy_proxy: float
    service_time_mean_ms: float
    service_time_std_ms: float

    def get_expected_service_time(self) -> float:
        return self.service_time_mean_ms


@dataclass
class Request:
    req_id: int
    tenant_id: int
    arrival_time: float  # ms
    slo_ms: float
    priority: int
    task_type: str = "image_classification"

    # Policy decisions
    model_chosen: Optional[ModelProfile] = None
    is_dropped: bool = False
    was_downgraded: bool = False
    drop_reason: str = ""
    predicted_finish_time: float = -1.0

    # Execution records
    start_time: float = -1.0  # ms
    finish_time: float = -1.0 # ms

    @property
    def deadline_time(self) -> float:
        return self.arrival_time + self.slo_ms
