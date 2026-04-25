import simpy
import random
from typing import Callable, Dict, Any
from sim.core import Request

class WorkloadGenerator:
    def __init__(self, env: simpy.Environment, config: Dict[str, Any], dispatcher_callback: Callable[[Request], None]):
        self.env = env
        self.config = config
        self.dispatcher_callback = dispatcher_callback

        self.req_id_counter = 0

        seed = config['simulation'].get('random_seed', 42)
        self.rand_gen = random.Random(seed)

        self.tenants = config['tenants']
        self.tenant_weights = [t['weight'] for t in self.tenants]
        self.tenant_ids = [t['id'] for t in self.tenants]

        self.arrival_process = config['simulation'].get('arrival_process', 'poisson')
        self.arrival_rate = config['simulation']['arrival_rate']  # requests per second
        self.onoff_cycle_ms = config['simulation'].get('onoff_cycle_ms', 1000.0)
        self.onoff_on_fraction = config['simulation'].get('onoff_on_fraction', 0.5)
        self.onoff_peak_rate_multiplier = config['simulation'].get('onoff_peak_rate_multiplier', 2.0)

    def run(self):
        while True:
            rate_ms = self.arrival_rate / 1000.0

            if self.arrival_process == 'poisson':
                inter_arrival_time_ms = self.rand_gen.expovariate(rate_ms)
            elif self.arrival_process == 'onoff':
                cycle_time = max(self.onoff_cycle_ms, 1.0)
                on_window = cycle_time * min(max(self.onoff_on_fraction, 0.01), 0.99)
                if (self.env.now % cycle_time) < on_window:
                    burst_rate_ms = rate_ms * max(self.onoff_peak_rate_multiplier, 1.0)
                    inter_arrival_time_ms = self.rand_gen.expovariate(burst_rate_ms)
                else:
                    inter_arrival_time_ms = cycle_time - (self.env.now % cycle_time)
            else:
                inter_arrival_time_ms = 1.0 / rate_ms

            yield self.env.timeout(inter_arrival_time_ms)

            tenant_id = self.rand_gen.choices(self.tenant_ids, weights=self.tenant_weights, k=1)[0]
            tenant_info = next(t for t in self.tenants if t['id'] == tenant_id)
            req = Request(
                req_id=self.req_id_counter,
                tenant_id=tenant_id,
                arrival_time=self.env.now,
                slo_ms=tenant_info['slo_ms'],
                priority=tenant_info['priority']
            )
            self.req_id_counter += 1
            self.dispatcher_callback(req)
