import copy
import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from sim.simulate import run_simulation


def _plot_sensitivity(df: pd.DataFrame, metric: str, title: str, ylabel: str, legend_key: str, output_path: str):
    plt.figure(figsize=(9, 6))
    for label in df[legend_key].unique():
        subset = df[df[legend_key] == label].sort_values("arrival_rate")
        plt.plot(subset["arrival_rate"], subset[metric], marker="o", label=str(label))

    plt.xlabel("Arrival Rate (req/s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title=legend_key)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _run_cases(base_config: dict, rates: list[int], seeds: list[int], cases: list[dict]) -> pd.DataFrame:
    rows = []
    for rate in rates:
        for case in cases:
            for seed in seeds:
                print(f"sensitivity: rate={rate}, seed={seed}, case={case['name']}")
                config = copy.deepcopy(base_config)
                config["simulation"]["arrival_rate"] = rate
                config["simulation"]["random_seed"] = seed
                config["policy"] = case["policy"]
                for sim_key, sim_value in case.get("simulation", {}).items():
                    config["simulation"][sim_key] = sim_value

                result = run_simulation(config)
                result["arrival_rate"] = rate
                result["seed"] = seed
                result["case"] = case["name"]
                for key, value in case["tags"].items():
                    result[key] = value
                rows.append(result)
    return pd.DataFrame(rows)


def run_sensitivity(base_config_path: str):
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    rates = [15, 20, 25, 30]
    seeds = [42, 43, 44]

    threshold_values = [80.0, 120.0, 160.0, 200.0]
    threshold_cases = [
        {
            "name": f"threshold_wait_ms={value}",
            "policy": {
                "scheduler": "EDF",
                "strategy": "threshold_downgrade",
                "fixed_model_name": "slow",
                "threshold_wait_ms": value,
                "threshold_queue_len": 4,
                "admission_control": False,
            },
            "tags": {"family": "threshold_wait_ms", "param_value": value},
        }
        for value in threshold_values
    ]

    utility_values = [0.0, 0.12, 0.5, 1.0, 2.0]
    utility_cases = [
        {
            "name": f"utility_latency_penalty={value}",
            "policy": {
                "scheduler": "EDF",
                "strategy": "slo_aware_utility",
                "fixed_model_name": "slow",
                "utility_latency_penalty": value,
                "utility_priority_bonus": 0.03,
                "admission_control": False,
            },
            "tags": {"family": "utility_latency_penalty", "param_value": value},
        }
        for value in utility_values
    ]

    pred_error_values = [-0.1, 0.0, 0.1, 0.2]
    pred_error_cases = [
        {
            "name": f"prediction_error_ratio={value:+.1f}",
            "policy": {
                "scheduler": "EDF",
                "strategy": "slo_aware_utility",
                "fixed_model_name": "slow",
                "utility_latency_penalty": 0.12,
                "utility_priority_bonus": 0.03,
                "prediction_error_ratio": value,
                "admission_control": True,
            },
            "tags": {"family": "prediction_error_ratio", "param_value": value},
        }
        for value in pred_error_values
    ]

    switch_overhead_values = [0.0, 5.0, 10.0, 20.0]
    switch_overhead_cases = [
        {
            "name": f"model_switch_overhead_ms={value:.0f}",
            "policy": {
                "scheduler": "EDF",
                "strategy": "slo_aware_utility",
                "fixed_model_name": "slow",
                "utility_latency_penalty": 0.12,
                "utility_priority_bonus": 0.03,
                "admission_control": True,
                "prediction_error_ratio": 0.0,
            },
            "simulation": {"model_switch_overhead_ms": value},
            "tags": {"family": "model_switch_overhead_ms", "param_value": value},
        }
        for value in switch_overhead_values
    ]

    threshold_raw = _run_cases(base_config, rates, seeds, threshold_cases)
    utility_raw = _run_cases(base_config, rates, seeds, utility_cases)
    pred_error_raw = _run_cases(base_config, rates, seeds, pred_error_cases)
    switch_overhead_raw = _run_cases(base_config, rates, seeds, switch_overhead_cases)

    threshold_raw.to_csv("results/phase4_threshold_sensitivity_raw.csv", index=False)
    utility_raw.to_csv("results/phase4_utility_sensitivity_raw.csv", index=False)
    pred_error_raw.to_csv("results/phase4_pred_error_raw.csv", index=False)
    switch_overhead_raw.to_csv("results/phase4_switch_overhead_raw.csv", index=False)

    metric_columns = [
        "overall_slo_miss_rate",
        "accepted_slo_miss_rate",
        "drop_rate",
        "accuracy_weighted_throughput",
        "mean_accuracy_completed",
        "downgrade_rate",
        "p99_latency_ms",
    ]
    threshold_agg = threshold_raw.groupby(["arrival_rate", "param_value"], as_index=False)[metric_columns].mean(numeric_only=True)
    utility_agg = utility_raw.groupby(["arrival_rate", "param_value"], as_index=False)[metric_columns].mean(numeric_only=True)
    pred_error_agg = pred_error_raw.groupby(["arrival_rate", "param_value"], as_index=False)[metric_columns].mean(
        numeric_only=True
    )
    switch_overhead_agg = switch_overhead_raw.groupby(["arrival_rate", "param_value"], as_index=False)[
        metric_columns
    ].mean(numeric_only=True)

    threshold_agg.to_csv("results/phase4_threshold_sensitivity_aggregated.csv", index=False)
    utility_agg.to_csv("results/phase4_utility_sensitivity_aggregated.csv", index=False)
    pred_error_agg.to_csv("results/phase4_pred_error_aggregated.csv", index=False)
    switch_overhead_agg.to_csv("results/phase4_switch_overhead_aggregated.csv", index=False)

    _plot_sensitivity(
        threshold_agg,
        "overall_slo_miss_rate",
        "Sensitivity: Threshold Wait vs. SLO Miss",
        "Overall SLO Miss Rate",
        "param_value",
        "figs/phase4_threshold_slo_miss.png",
    )
    _plot_sensitivity(
        threshold_agg,
        "accuracy_weighted_throughput",
        "Sensitivity: Threshold Wait vs. Accuracy-Weighted Throughput",
        "Accuracy-Weighted Throughput",
        "param_value",
        "figs/phase4_threshold_acc_throughput.png",
    )
    _plot_sensitivity(
        utility_agg,
        "overall_slo_miss_rate",
        "Sensitivity: Utility Penalty vs. SLO Miss",
        "Overall SLO Miss Rate",
        "param_value",
        "figs/phase4_utility_slo_miss.png",
    )
    _plot_sensitivity(
        utility_agg,
        "accuracy_weighted_throughput",
        "Sensitivity: Utility Penalty vs. Accuracy-Weighted Throughput",
        "Accuracy-Weighted Throughput",
        "param_value",
        "figs/phase4_utility_acc_throughput.png",
    )
    _plot_sensitivity(
        pred_error_agg,
        "overall_slo_miss_rate",
        "Sensitivity: Prediction Error vs. SLO Miss",
        "Overall SLO Miss Rate",
        "param_value",
        "figs/phase4_pred_error_slo_miss.png",
    )
    _plot_sensitivity(
        pred_error_agg,
        "drop_rate",
        "Sensitivity: Prediction Error vs. Drop Rate",
        "Drop Rate",
        "param_value",
        "figs/phase4_pred_error_drop_rate.png",
    )
    _plot_sensitivity(
        switch_overhead_agg,
        "overall_slo_miss_rate",
        "Sensitivity: Switch Overhead vs. SLO Miss",
        "Overall SLO Miss Rate",
        "param_value",
        "figs/phase4_switch_overhead_slo_miss.png",
    )
    _plot_sensitivity(
        switch_overhead_agg,
        "accuracy_weighted_throughput",
        "Sensitivity: Switch Overhead vs. Accuracy-Weighted Throughput",
        "Accuracy-Weighted Throughput",
        "param_value",
        "figs/phase4_switch_overhead_acc_throughput.png",
    )

    print("Sensitivity analysis completed. Results saved to 'results/' and 'figs/'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    run_sensitivity(args.config)
