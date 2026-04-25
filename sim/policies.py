from typing import Any, Dict, List, Optional, Tuple

from sim.core import ModelProfile, Request


class Policy:
    def __init__(self, config: Dict[str, Any], model_zoo: List[ModelProfile]):
        self.config = config
        self.policy_config = config["policy"]
        self.model_zoo = sorted(model_zoo, key=lambda m: m.accuracy_proxy, reverse=True)
        self.fastest_models = sorted(model_zoo, key=lambda m: m.get_expected_service_time())

        self.scheduler = self.policy_config.get("scheduler", "FIFO")
        self.strategy = self.policy_config.get("strategy", "fixed_high_accuracy")
        self.use_admission = self.policy_config.get("admission_control", False)

        self.threshold_wait_ms = self.policy_config.get("threshold_wait_ms", 75.0)
        self.threshold_queue_len = self.policy_config.get("threshold_queue_len", 3)
        self.utility_latency_penalty = self.policy_config.get("utility_latency_penalty", 0.1)
        self.utility_priority_bonus = self.policy_config.get("utility_priority_bonus", 0.02)
        self.prediction_error_ratio = self.policy_config.get("prediction_error_ratio", 0.0)

        fixed_model_name = self.policy_config.get("fixed_model_name", self.model_zoo[0].name)
        self.fixed_model = self._find_model_by_name(fixed_model_name) or self.model_zoo[0]

    def _find_model_by_name(self, model_name: str) -> Optional[ModelProfile]:
        for model in self.model_zoo:
            if model.name == model_name:
                return model
        return None

    def _queue_sort_key(self, req: Request):
        if self.scheduler == "FIFO":
            return (req.arrival_time, req.req_id)
        if self.scheduler == "EDF":
            return (req.deadline_time, req.arrival_time, req.req_id)
        if self.scheduler == "Priority":
            return (req.priority, req.arrival_time, req.req_id)
        return (req.arrival_time, req.req_id)

    def _get_predicted_service_time(self, model: ModelProfile) -> float:
        multiplier = max(0.01, 1.0 + self.prediction_error_ratio)
        return model.get_expected_service_time() * multiplier

    def _predict_finish_for_candidate(
        self, req: Request, state: Dict[str, Any], candidate_model: ModelProfile
    ) -> float:
        simulated_req = Request(
            req_id=req.req_id,
            tenant_id=req.tenant_id,
            arrival_time=req.arrival_time,
            slo_ms=req.slo_ms,
            priority=req.priority,
            task_type=req.task_type,
            model_chosen=candidate_model,
        )

        server_finish_time = max(state["current_time"], state["server_finish_time"])
        pred_time = server_finish_time

        simulated_queue = list(state["queue"]) + [simulated_req]
        simulated_queue.sort(key=self._queue_sort_key)

        for queued_req in simulated_queue:
            pred_time += self._get_predicted_service_time(queued_req.model_chosen)
            if queued_req.req_id == simulated_req.req_id:
                return pred_time

        return pred_time

    def _estimate_pred_wait(self, req: Request, state: Dict[str, Any], candidate_model: ModelProfile) -> float:
        pred_finish = self._predict_finish_for_candidate(req, state, candidate_model)
        return pred_finish - state["current_time"] - self._get_predicted_service_time(candidate_model)

    def _predict_finish(self, now: float, pred_wait: float, model: ModelProfile) -> float:
        return now + pred_wait + self._get_predicted_service_time(model)

    def _pick_threshold_model(self, pred_wait: float, queue_len: int) -> ModelProfile:
        if pred_wait >= self.threshold_wait_ms or queue_len >= self.threshold_queue_len:
            return self.fastest_models[0]
        return self.fixed_model

    def _utility_score(self, req: Request, pred_finish: float, model: ModelProfile) -> float:
        latency_ratio = (pred_finish - req.arrival_time) / max(req.slo_ms, 1.0)
        priority_bonus = self.utility_priority_bonus / max(req.priority, 1)
        return model.accuracy_proxy - (self.utility_latency_penalty * latency_ratio) + priority_bonus

    def _pick_feasible_utility_model(self, req: Request, state: Dict[str, Any]) -> ModelProfile:
        feasible_candidates = []
        for model in self.model_zoo:
            pred_finish = self._predict_finish_for_candidate(req, state, model)
            if pred_finish <= req.deadline_time:
                score = self._utility_score(req, pred_finish, model)
                feasible_candidates.append((score, model))

        if feasible_candidates:
            feasible_candidates.sort(key=lambda item: item[0], reverse=True)
            return feasible_candidates[0][1]

        return self.fastest_models[0]

    def _pick_fastest_feasible_model(self, req: Request, state: Dict[str, Any]) -> ModelProfile:
        feasible_candidates = []
        for model in self.fastest_models:
            pred_finish = self._predict_finish_for_candidate(req, state, model)
            if pred_finish <= req.deadline_time:
                feasible_candidates.append(model)

        if feasible_candidates:
            return feasible_candidates[0]

        return self.fastest_models[0]

    def _choose_model(
        self, req: Request, state: Dict[str, Any], _now: float, pred_wait: float, queue_len: int
    ) -> ModelProfile:
        if self.strategy == "fixed_high_accuracy":
            return self.fixed_model
        if self.strategy == "threshold_downgrade":
            return self._pick_threshold_model(pred_wait, queue_len)
        if self.strategy == "fastest_feasible":
            return self._pick_fastest_feasible_model(req, state)
        if self.strategy == "slo_aware_utility":
            return self._pick_feasible_utility_model(req, state)
        return self.fixed_model

    def decide_on_arrival(self, req: Request, state: Dict[str, Any]) -> Tuple[bool, Request]:
        now = state["current_time"]
        queue = state["queue"]
        pred_wait = self._estimate_pred_wait(req, state, self.fixed_model)

        chosen_model = self._choose_model(req, state, now, pred_wait, len(queue))
        pred_wait = self._estimate_pred_wait(req, state, chosen_model)
        pred_finish = self._predict_finish(now, pred_wait, chosen_model)

        req.model_chosen = chosen_model
        req.predicted_finish_time = pred_finish
        req.was_downgraded = chosen_model.name != self.fixed_model.name

        if self.use_admission and pred_finish > req.deadline_time:
            req.is_dropped = True
            req.drop_reason = "deadline_infeasible"
            return False, req

        return True, req

    def pick_next(self, queue: List[Request], _now: float) -> Optional[Request]:
        if not queue:
            return None

        if self.scheduler == "FIFO":
            idx = min(range(len(queue)), key=lambda i: queue[i].arrival_time)
        elif self.scheduler == "EDF":
            idx = min(range(len(queue)), key=lambda i: queue[i].deadline_time)
        elif self.scheduler == "Priority":
            idx = min(range(len(queue)), key=lambda i: (queue[i].priority, queue[i].arrival_time))
        else:
            idx = 0

        return queue.pop(idx)
