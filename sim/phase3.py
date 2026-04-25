import copy
import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from sim.simulate import run_simulation


def _plot_lines(df: pd.DataFrame, x_key: str, y_key: str, group_key: str, title: str, ylabel: str, output_path: str):
    plt.figure(figsize=(9, 6))
    for name in df[group_key].unique():
        subset = df[df[group_key] == name].sort_values(x_key)
        plt.plot(subset[x_key], subset[y_key], marker="o", label=name)

    plt.xlabel("Arrival Rate (req/s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _run_matrix(base_config: dict, rates: list[int], seeds: list[int], experiment_name: str, cases: list[dict]) -> pd.DataFrame:
    rows = []
    for rate in rates:
        for case in cases:
            for seed in seeds:
                print(f"{experiment_name}: rate={rate}, seed={seed}, case={case['name']}")
                config = copy.deepcopy(base_config)
                config["simulation"]["arrival_rate"] = rate
                config["simulation"]["random_seed"] = seed
                config["simulation"].update(case.get("simulation", {}))
                config["policy"] = case["policy"]
                if "tenants" in case:
                    config["tenants"] = copy.deepcopy(case["tenants"])

                result = run_simulation(config)
                tenant_miss = result.get("per_tenant_slo_miss_rate", {})
                result["tenant0_slo_miss_rate"] = tenant_miss.get(0, 0.0)
                result["tenant1_slo_miss_rate"] = tenant_miss.get(1, 0.0)
                result["arrival_rate"] = rate
                result["seed"] = seed
                result["case"] = case["name"]
                for key, value in case.get("tags", {}).items():
                    result[key] = value
                rows.append(result)
    return pd.DataFrame(rows)


def run_phase3(base_config_path: str):
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    rates = [10, 15, 20, 25, 30]
    seeds = [42, 43, 44]

    burstiness_cases = [
        {
            "name": "Poisson | EDF + Fixed",
            "simulation": {"arrival_process": "poisson"},
            "policy": {
                "scheduler": "EDF",
                "strategy": "fixed_high_accuracy",
                "fixed_model_name": "slow",
                "admission_control": False,
            },
            "tags": {"experiment": "burstiness", "workload": "Poisson", "policy_name": "EDF + Fixed"},
        },
        {
            "name": "Poisson | EDF + Threshold Downgrade",
            "simulation": {"arrival_process": "poisson"},
            "policy": {
                "scheduler": "EDF",
                "strategy": "threshold_downgrade",
                "fixed_model_name": "slow",
                "threshold_wait_ms": 120.0,
                "threshold_queue_len": 4,
                "admission_control": False,
            },
            "tags": {"experiment": "burstiness", "workload": "Poisson", "policy_name": "EDF + Threshold Downgrade"},
        },
        {
            "name": "Poisson | Proposed",
            "simulation": {"arrival_process": "poisson"},
            "policy": {
                "scheduler": "EDF",
                "strategy": "slo_aware_utility",
                "fixed_model_name": "slow",
                "utility_latency_penalty": 0.12,
                "utility_priority_bonus": 0.03,
                "admission_control": False,
            },
            "tags": {"experiment": "burstiness", "workload": "Poisson", "policy_name": "Proposed"},
        },
        {
            "name": "ON-OFF | EDF + Fixed",
            "simulation": {
                "arrival_process": "onoff",
                "onoff_cycle_ms": 1000.0,
                "onoff_on_fraction": 0.25,
                "onoff_peak_rate_multiplier": 4.0,
            },
            "policy": {
                "scheduler": "EDF",
                "strategy": "fixed_high_accuracy",
                "fixed_model_name": "slow",
                "admission_control": False,
            },
            "tags": {"experiment": "burstiness", "workload": "ON-OFF", "policy_name": "EDF + Fixed"},
        },
        {
            "name": "ON-OFF | EDF + Threshold Downgrade",
            "simulation": {
                "arrival_process": "onoff",
                "onoff_cycle_ms": 1000.0,
                "onoff_on_fraction": 0.25,
                "onoff_peak_rate_multiplier": 4.0,
            },
            "policy": {
                "scheduler": "EDF",
                "strategy": "threshold_downgrade",
                "fixed_model_name": "slow",
                "threshold_wait_ms": 120.0,
                "threshold_queue_len": 4,
                "admission_control": False,
            },
            "tags": {"experiment": "burstiness", "workload": "ON-OFF", "policy_name": "EDF + Threshold Downgrade"},
        },
        {
            "name": "ON-OFF | Proposed",
            "simulation": {
                "arrival_process": "onoff",
                "onoff_cycle_ms": 1000.0,
                "onoff_on_fraction": 0.25,
                "onoff_peak_rate_multiplier": 4.0,
            },
            "policy": {
                "scheduler": "EDF",
                "strategy": "slo_aware_utility",
                "fixed_model_name": "slow",
                "utility_latency_penalty": 0.12,
                "utility_priority_bonus": 0.03,
                "admission_control": False,
            },
            "tags": {"experiment": "burstiness", "workload": "ON-OFF", "policy_name": "Proposed"},
        },
    ]

    ablation_cases = [
        {
            "name": "Admission Only + Fixed Model",
            "policy": {
                "scheduler": "EDF",
                "strategy": "fixed_high_accuracy",
                "fixed_model_name": "slow",
                "admission_control": True,
            },
            "tags": {"experiment": "ablation", "policy_name": "Admission Only + Fixed"},
        },
        {
            "name": "Threshold Downgrade",
            "policy": {
                "scheduler": "EDF",
                "strategy": "threshold_downgrade",
                "fixed_model_name": "slow",
                "threshold_wait_ms": 120.0,
                "threshold_queue_len": 4,
                "admission_control": False,
            },
            "tags": {"experiment": "ablation", "policy_name": "Threshold Downgrade"},
        },
        {
            "name": "Ablation: Fastest Feasible",
            "policy": {
                "scheduler": "EDF",
                "strategy": "fastest_feasible",
                "fixed_model_name": "slow",
                "admission_control": False,
            },
            "tags": {"experiment": "ablation", "policy_name": "Fastest Feasible"},
        },
        {
            "name": "Proposed: Feasible Utility",
            "policy": {
                "scheduler": "EDF",
                "strategy": "slo_aware_utility",
                "fixed_model_name": "slow",
                "utility_latency_penalty": 0.12,
                "utility_priority_bonus": 0.03,
                "admission_control": False,
            },
            "tags": {"experiment": "ablation", "policy_name": "Proposed"},
        },
        {
            "name": "Proposed: Feasible Utility + Admission",
            "policy": {
                "scheduler": "EDF",
                "strategy": "slo_aware_utility",
                "fixed_model_name": "slow",
                "utility_latency_penalty": 0.12,
                "utility_priority_bonus": 0.03,
                "admission_control": True,
            },
            "tags": {"experiment": "ablation", "policy_name": "Proposed + Admission"},
        },
    ]

    heterogeneity_cases = [
        {
            "name": "Balanced | Threshold",
            "policy": {
                "scheduler": "EDF",
                "strategy": "threshold_downgrade",
                "fixed_model_name": "slow",
                "threshold_wait_ms": 120.0,
                "threshold_queue_len": 4,
                "admission_control": False,
            },
            "tags": {"experiment": "heterogeneity", "scenario": "Balanced", "policy_name": "Threshold Downgrade"},
        },
        {
            "name": "Balanced | Proposed",
            "policy": {
                "scheduler": "EDF",
                "strategy": "slo_aware_utility",
                "fixed_model_name": "slow",
                "utility_latency_penalty": 0.12,
                "utility_priority_bonus": 0.03,
                "admission_control": False,
            },
            "tags": {"experiment": "heterogeneity", "scenario": "Balanced", "policy_name": "Proposed"},
        },
        {
            "name": "Strict-Heavy | Threshold",
            "tenants": [
                {"id": 0, "weight": 0.7, "slo_ms": 80, "priority": 1},
                {"id": 1, "weight": 0.3, "slo_ms": 350, "priority": 3},
            ],
            "policy": {
                "scheduler": "EDF",
                "strategy": "threshold_downgrade",
                "fixed_model_name": "slow",
                "threshold_wait_ms": 120.0,
                "threshold_queue_len": 4,
                "admission_control": False,
            },
            "tags": {"experiment": "heterogeneity", "scenario": "Strict-Heavy", "policy_name": "Threshold Downgrade"},
        },
        {
            "name": "Strict-Heavy | Proposed",
            "tenants": [
                {"id": 0, "weight": 0.7, "slo_ms": 80, "priority": 1},
                {"id": 1, "weight": 0.3, "slo_ms": 350, "priority": 3},
            ],
            "policy": {
                "scheduler": "EDF",
                "strategy": "slo_aware_utility",
                "fixed_model_name": "slow",
                "utility_latency_penalty": 0.12,
                "utility_priority_bonus": 0.03,
                "admission_control": False,
            },
            "tags": {"experiment": "heterogeneity", "scenario": "Strict-Heavy", "policy_name": "Proposed"},
        },
    ]

    burstiness_raw = _run_matrix(base_config, rates, seeds, "burstiness", burstiness_cases)
    ablation_raw = _run_matrix(base_config, rates, seeds, "ablation", ablation_cases)
    heterogeneity_raw = _run_matrix(base_config, rates, seeds, "heterogeneity", heterogeneity_cases)

    burstiness_raw.to_csv("results/phase3_burstiness_raw.csv", index=False)
    ablation_raw.to_csv("results/phase3_ablation_raw.csv", index=False)
    heterogeneity_raw.to_csv("results/phase3_heterogeneity_raw.csv", index=False)

    metric_columns = [
        "overall_slo_miss_rate",
        "accepted_slo_miss_rate",
        "p99_latency_ms",
        "accuracy_weighted_throughput",
        "mean_accuracy_completed",
        "downgrade_rate",
        "fairness_jain_index",
        "completion_rate",
        "tenant0_slo_miss_rate",
        "tenant1_slo_miss_rate",
    ]

    burstiness_agg = burstiness_raw.groupby(["arrival_rate", "case", "workload", "policy_name"], as_index=False)[metric_columns].mean(numeric_only=True)
    ablation_agg = ablation_raw.groupby(["arrival_rate", "case", "policy_name"], as_index=False)[metric_columns].mean(numeric_only=True)
    heterogeneity_agg = heterogeneity_raw.groupby(["arrival_rate", "case", "scenario", "policy_name"], as_index=False)[metric_columns].mean(numeric_only=True)

    burstiness_agg.to_csv("results/phase3_burstiness_aggregated.csv", index=False)
    ablation_agg.to_csv("results/phase3_ablation_aggregated.csv", index=False)
    heterogeneity_agg.to_csv("results/phase3_heterogeneity_aggregated.csv", index=False)

    _plot_lines(
        burstiness_agg,
        "arrival_rate",
        "overall_slo_miss_rate",
        "case",
        "Burstiness: SLO Miss Rate vs. Load",
        "Overall SLO Miss Rate",
        "figs/phase3_burstiness_slo_miss.png",
    )
    _plot_lines(
        burstiness_agg,
        "arrival_rate",
        "p99_latency_ms",
        "case",
        "Burstiness: P99 Latency vs. Load",
        "P99 Latency of Completed Requests (ms)",
        "figs/phase3_burstiness_p99.png",
    )
    _plot_lines(
        ablation_agg,
        "arrival_rate",
        "overall_slo_miss_rate",
        "policy_name",
        "Ablation: SLO Miss Rate vs. Load",
        "Overall SLO Miss Rate",
        "figs/phase3_ablation_slo_miss.png",
    )
    _plot_lines(
        ablation_agg,
        "arrival_rate",
        "accuracy_weighted_throughput",
        "policy_name",
        "Ablation: Accuracy-Weighted Throughput vs. Load",
        "Accuracy-Weighted Throughput",
        "figs/phase3_ablation_acc_throughput.png",
    )
    _plot_lines(
        heterogeneity_agg,
        "arrival_rate",
        "tenant0_slo_miss_rate",
        "case",
        "Heterogeneity: Strict Tenant SLO Miss vs. Load",
        "Tenant 0 SLO Miss Rate",
        "figs/phase3_heterogeneity_strict_tenant_miss.png",
    )
    _plot_lines(
        heterogeneity_agg,
        "arrival_rate",
        "fairness_jain_index",
        "case",
        "Heterogeneity: Fairness vs. Load",
        "Jain Fairness Index",
        "figs/phase3_heterogeneity_fairness.png",
    )

    print("Phase 3 experiments completed. Results saved to 'results/' and 'figs/'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    run_phase3(args.config)
