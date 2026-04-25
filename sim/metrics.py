import numpy as np
from typing import List, Dict, Any
from sim.core import Request

class MetricsCollector:
    def __init__(self, simulation_time_ms: float, num_tenants: int):
        self.simulation_time_ms = simulation_time_ms
        self.num_tenants = num_tenants
        self.arrived_requests: List[Request] = []
        self.completed_requests: List[Request] = []
        self.dropped_requests: List[Request] = []
        self.unfinished_requests: List[Request] = []

    def add_arrival(self, req: Request):
        self.arrived_requests.append(req)

    def add_completed(self, req: Request):
        self.completed_requests.append(req)

    def add_dropped(self, req: Request):
        self.dropped_requests.append(req)

    def add_unfinished(self, req: Request):
        self.unfinished_requests.append(req)

    def calculate_metrics(self) -> Dict[str, Any]:
        total_reqs = len(self.arrived_requests)

        if total_reqs == 0:
            return {}

        simulation_time_s = self.simulation_time_ms / 1000.0

        accepted_requests = total_reqs - len(self.dropped_requests)
        metrics = {
            "total_requests": total_reqs,
            "accepted_requests": accepted_requests,
            "completed_requests": len(self.completed_requests),
            "dropped_requests": len(self.dropped_requests),
            "unfinished_requests": len(self.unfinished_requests),
            "drop_rate": len(self.dropped_requests) / total_reqs,
            "throughput": len(self.completed_requests) / simulation_time_s,
        }

        latencies = [r.finish_time - r.arrival_time for r in self.completed_requests]
        accuracies = [r.model_chosen.accuracy_proxy for r in self.completed_requests]
        completed_slo_misses = [(r.finish_time - r.arrival_time) > r.slo_ms for r in self.completed_requests]
        num_completed_misses = int(sum(completed_slo_misses))
        total_slo_misses = num_completed_misses + len(self.dropped_requests) + len(self.unfinished_requests)

        metrics["overall_slo_miss_rate"] = total_slo_misses / total_reqs
        metrics["accepted_slo_miss_rate"] = (
            (num_completed_misses + len(self.unfinished_requests)) / accepted_requests
            if accepted_requests > 0
            else 0.0
        )
        metrics["completion_rate"] = (
            len(self.completed_requests) / accepted_requests if accepted_requests > 0 else 0.0
        )
        metrics["accuracy_weighted_throughput"] = sum(accuracies) / simulation_time_s if accuracies else 0.0
        metrics["mean_accuracy_completed"] = float(np.mean(accuracies)) if accuracies else 0.0

        accepted_non_dropped = self.completed_requests + self.unfinished_requests
        metrics["downgrade_rate"] = (
            sum(1 for req in accepted_non_dropped if req.was_downgraded) / len(accepted_non_dropped)
            if accepted_non_dropped
            else 0.0
        )

        if latencies:
            metrics["p50_latency_ms"] = float(np.percentile(latencies, 50))
            metrics["p95_latency_ms"] = float(np.percentile(latencies, 95))
            metrics["p99_latency_ms"] = float(np.percentile(latencies, 99))
        else:
            metrics["p50_latency_ms"] = np.nan
            metrics["p95_latency_ms"] = np.nan
            metrics["p99_latency_ms"] = np.nan

        tenant_slo_miss = [0] * self.num_tenants
        tenant_total = [0] * self.num_tenants

        for r in self.dropped_requests:
            tenant_total[r.tenant_id] += 1
            tenant_slo_miss[r.tenant_id] += 1

        for r in self.unfinished_requests:
            tenant_total[r.tenant_id] += 1
            tenant_slo_miss[r.tenant_id] += 1

        for r in self.completed_requests:
            tenant_total[r.tenant_id] += 1
            if (r.finish_time - r.arrival_time) > r.slo_ms:
                tenant_slo_miss[r.tenant_id] += 1

        metrics["per_tenant_slo_miss_rate"] = {
            tid: (tenant_slo_miss[tid] / tenant_total[tid] if tenant_total[tid] > 0 else 0.0)
            for tid in range(self.num_tenants)
        }

        satisfaction_rates = []
        for tid in range(self.num_tenants):
            if tenant_total[tid] > 0:
                sat = 1.0 - (tenant_slo_miss[tid] / tenant_total[tid])
                satisfaction_rates.append(sat)

        if satisfaction_rates:
            sum_sq = sum(satisfaction_rates) ** 2
            sq_sum = sum(x ** 2 for x in satisfaction_rates)
            metrics["fairness_jain_index"] = sum_sq / (self.num_tenants * sq_sum) if sq_sum > 0 else 0.0
        else:
            metrics["fairness_jain_index"] = 0.0

        return metrics
