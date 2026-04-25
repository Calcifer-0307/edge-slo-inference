import simpy
import random
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from sim.core import ModelProfile, Request
from sim.workload import WorkloadGenerator
from sim.policies import Policy
from sim.metrics import MetricsCollector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class InferenceServer:
    def __init__(
        self,
        env: simpy.Environment,
        policy: Policy,
        metrics: MetricsCollector,
        seed: int,
        simulation_config: Dict[str, Any],
    ):
        self.env = env
        self.policy = policy
        self.metrics = metrics
        self.queue = []
        self.server_idle = True
        self.current_req_finish_time = 0.0
        self.current_req = None
        self.rand_gen = random.Random(seed + 1)
        self.model_switch_overhead_ms = simulation_config.get("model_switch_overhead_ms", 0.0)
        self.last_model_name = None
        self.action = env.process(self.run())

    def run(self):
        while True:
            if not self.queue:
                self.server_idle = True
                try:
                    # Sleep indefinitely until a request arrives and interrupts
                    yield self.env.timeout(1e9)
                except simpy.Interrupt:
                    pass
            
            self.server_idle = False
            req = self.policy.pick_next(self.queue, self.env.now)
            if req is None:
                continue

            self.current_req = req
            req.start_time = self.env.now

            mean = req.model_chosen.service_time_mean_ms
            std = req.model_chosen.service_time_std_ms
            actual_service_time = max(1.0, self.rand_gen.normalvariate(mean, std))
            if self.last_model_name is not None and self.last_model_name != req.model_chosen.name:
                actual_service_time += self.model_switch_overhead_ms

            self.current_req_finish_time = self.env.now + actual_service_time
            yield self.env.timeout(actual_service_time)

            req.finish_time = self.env.now
            self.metrics.add_completed(req)
            self.last_model_name = req.model_chosen.name
            self.current_req = None

    def handle_arrival(self, req: Request):
        self.metrics.add_arrival(req)
        state = {
            'current_time': self.env.now,
            'server_finish_time': self.current_req_finish_time if not self.server_idle else self.env.now,
            'queue': self.queue
        }

        accepted, processed_req = self.policy.decide_on_arrival(req, state)

        if accepted:
            self.queue.append(processed_req)
            if self.server_idle:
                self.action.interrupt()
        else:
            self.metrics.add_dropped(processed_req)

    def flush_unfinished_requests(self):
        if self.current_req is not None and self.current_req.finish_time < 0 and not self.current_req.is_dropped:
            self.metrics.add_unfinished(self.current_req)
            self.current_req = None

        for queued_req in self.queue:
            if queued_req.finish_time < 0 and not queued_req.is_dropped:
                self.metrics.add_unfinished(queued_req)
        self.queue.clear()

def load_model_zoo(config_dict: Dict[str, Any]) -> List[ModelProfile]:
    service_time_csv = DATA_DIR / "service_time.csv"
    accuracy_csv = DATA_DIR / "accuracy.csv"
    real_stats = {}
    real_accuracy = {}

    if service_time_csv.exists():
        df = pd.read_csv(service_time_csv)
        for _, row in df.iterrows():
            real_stats[row['name']] = {
                'mean': row['service_time_mean_ms'],
                'std': row['service_time_std_ms']
            }

    if accuracy_csv.exists():
        df = pd.read_csv(accuracy_csv)
        for _, row in df.iterrows():
            real_accuracy[row["name"]] = row["accuracy_proxy"]

    model_zoo = []
    for m in config_dict['model_zoo']:
        mean_ms = real_stats[m['name']]['mean'] if m['name'] in real_stats else m['service_time_mean_ms']
        std_ms = real_stats[m['name']]['std'] if m['name'] in real_stats else m['service_time_std_ms']
        accuracy_proxy = real_accuracy[m["name"]] if m["name"] in real_accuracy else m["accuracy_proxy"]

        model_zoo.append(
            ModelProfile(
                name=m['name'],
                accuracy_proxy=accuracy_proxy,
                service_time_mean_ms=mean_ms,
                service_time_std_ms=std_ms
            )
        )
    return model_zoo

def run_simulation(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    seed = config_dict['simulation'].get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    
    model_zoo = load_model_zoo(config_dict)
    
    env = simpy.Environment()
    policy = Policy(config_dict, model_zoo)
    metrics = MetricsCollector(config_dict['simulation']['time_s'] * 1000.0, len(config_dict['tenants']))
    
    server = InferenceServer(env, policy, metrics, seed, config_dict["simulation"])
    workload_gen = WorkloadGenerator(env, config_dict, server.handle_arrival)
    
    env.process(workload_gen.run())
    
    env.run(until=config_dict['simulation']['time_s'] * 1000.0)
    server.flush_unfinished_requests()

    return metrics.calculate_metrics()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    res = run_simulation(config)
    print("Simulation Results:")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
