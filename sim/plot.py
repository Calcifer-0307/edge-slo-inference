import matplotlib.pyplot as plt
import copy
import math
import os
import pandas as pd
import yaml

from sim.simulate import run_simulation


def get_phase2_policies():
    return [
        {
            "name": "FIFO + Fixed",
            "config": {
                "scheduler": "FIFO",
                "strategy": "fixed_high_accuracy",
                "fixed_model_name": "slow",
                "admission_control": False,
            },
        },
        {
            "name": "EDF + Fixed",
            "config": {
                "scheduler": "EDF",
                "strategy": "fixed_high_accuracy",
                "fixed_model_name": "slow",
                "admission_control": False,
            },
        },
        {
            "name": "EDF + Admission Only + Fixed Model",
            "config": {
                "scheduler": "EDF",
                "strategy": "fixed_high_accuracy",
                "fixed_model_name": "slow",
                "admission_control": True,
            },
        },
        {
            "name": "EDF + Threshold Downgrade",
            "config": {
                "scheduler": "EDF",
                "strategy": "threshold_downgrade",
                "fixed_model_name": "slow",
                "threshold_wait_ms": 120.0,
                "threshold_queue_len": 4,
                "admission_control": False,
            },
        },
        {
            "name": "EDF + Threshold Downgrade + Admission",
            "config": {
                "scheduler": "EDF",
                "strategy": "threshold_downgrade",
                "fixed_model_name": "slow",
                "threshold_wait_ms": 120.0,
                "threshold_queue_len": 4,
                "admission_control": True,
            },
        },
        {
            "name": "Proposed: EDF + Feasible Utility",
            "config": {
                "scheduler": "EDF",
                "strategy": "slo_aware_utility",
                "fixed_model_name": "slow",
                "utility_latency_penalty": 0.12,
                "utility_priority_bonus": 0.03,
                "admission_control": False,
            },
        },
        {
            "name": "Proposed: EDF + Feasible Utility + Admission",
            "config": {
                "scheduler": "EDF",
                "strategy": "slo_aware_utility",
                "fixed_model_name": "slow",
                "utility_latency_penalty": 0.12,
                "utility_priority_bonus": 0.03,
                "admission_control": True,
            },
        },
    ]


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: str):
    plt.figure(figsize=(8, 6))
    for pol_name in df["policy"].unique():
        subset = df[df["policy"] == pol_name].sort_values("arrival_rate")
        plt.plot(subset["arrival_rate"], subset[metric], marker="o", label=pol_name)

    plt.xlabel("Arrival Rate (req/s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_metric_with_ci(
    df: pd.DataFrame, mean_metric: str, ci_metric: str, ylabel: str, title: str, output_path: str
):
    plt.figure(figsize=(8, 6))
    for pol_name in df["policy"].unique():
        subset = df[df["policy"] == pol_name].sort_values("arrival_rate")
        plt.errorbar(
            subset["arrival_rate"],
            subset[mean_metric],
            yerr=subset[ci_metric],
            marker="o",
            capsize=4,
            label=pol_name,
        )

    plt.xlabel("Arrival Rate (req/s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _build_stats_df(raw_df: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    grouped = raw_df.groupby(group_columns)
    rows = []

    for group_key, group_df in grouped:
        row = {}
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for column_name, value in zip(group_columns, group_key):
            row[column_name] = value

        count = len(group_df)
        row["sample_count"] = count

        for metric in metric_columns:
            mean_value = group_df[metric].mean()
            std_value = group_df[metric].std(ddof=1)
            if pd.isna(std_value):
                std_value = 0.0
            ci95_value = 1.96 * (std_value / math.sqrt(count)) if count > 0 else 0.0

            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_ci95"] = ci95_value

        rows.append(row)

    return pd.DataFrame(rows)


def run_experiment_sweep(base_config_path: str):
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    rates = [5, 10, 15, 20, 25, 30]
    seeds = [42, 43, 44]
    policies = get_phase2_policies()

    raw_results = []

    for rate in rates:
        for pol in policies:
            for seed in seeds:
                print(f"Running sweep: Rate={rate}, Seed={seed}, Policy={pol['name']}")
                config = copy.deepcopy(base_config)
                config["simulation"]["arrival_rate"] = rate
                config["simulation"]["random_seed"] = seed
                config["policy"] = pol["config"]

                res = run_simulation(config)
                res["arrival_rate"] = rate
                res["policy"] = pol["name"]
                res["seed"] = seed
                raw_results.append(res)

    raw_df = pd.DataFrame(raw_results)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    raw_df.to_csv("results/phase2_raw_results.csv", index=False)

    metric_columns = [
        "total_requests",
        "accepted_requests",
        "completed_requests",
        "dropped_requests",
        "unfinished_requests",
        "drop_rate",
        "throughput",
        "overall_slo_miss_rate",
        "accepted_slo_miss_rate",
        "completion_rate",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "accuracy_weighted_throughput",
        "mean_accuracy_completed",
        "downgrade_rate",
        "fairness_jain_index",
    ]
    aggregated_df = raw_df.groupby(["arrival_rate", "policy"], as_index=False)[metric_columns].mean(numeric_only=True)
    aggregated_df.to_csv("results/phase2_aggregated_results.csv", index=False)
    stats_df = _build_stats_df(raw_df, ["arrival_rate", "policy"], metric_columns)
    stats_df.to_csv("results/phase2_stats_results.csv", index=False)

    _plot_metric(
        aggregated_df,
        "overall_slo_miss_rate",
        "Overall SLO Miss Rate",
        "SLO Miss Rate vs. Load",
        "figs/slo_miss_vs_load.png",
    )
    _plot_metric(
        aggregated_df,
        "accepted_slo_miss_rate",
        "Accepted-Request SLO Miss Rate",
        "Accepted SLO Miss Rate vs. Load",
        "figs/accepted_slo_miss_vs_load.png",
    )
    _plot_metric(
        aggregated_df,
        "drop_rate",
        "Drop Rate",
        "Drop Rate vs. Load",
        "figs/drop_rate_vs_load.png",
    )
    _plot_metric(
        aggregated_df,
        "accuracy_weighted_throughput",
        "Accuracy-Weighted Throughput",
        "Accuracy-Weighted Throughput vs. Load",
        "figs/acc_throughput_vs_load.png",
    )
    _plot_metric(
        aggregated_df,
        "p99_latency_ms",
        "P99 Latency of Completed Requests (ms)",
        "P99 Latency vs. Load",
        "figs/p99_latency_vs_load.png",
    )
    _plot_metric(
        aggregated_df,
        "fairness_jain_index",
        "Jain Fairness Index",
        "Fairness vs. Load",
        "figs/fairness_vs_load.png",
    )
    _plot_metric_with_ci(
        stats_df,
        "overall_slo_miss_rate_mean",
        "overall_slo_miss_rate_ci95",
        "Overall SLO Miss Rate",
        "SLO Miss Rate vs. Load (95% CI)",
        "figs/slo_miss_vs_load_ci.png",
    )
    _plot_metric_with_ci(
        stats_df,
        "accepted_slo_miss_rate_mean",
        "accepted_slo_miss_rate_ci95",
        "Accepted-Request SLO Miss Rate",
        "Accepted SLO Miss Rate vs. Load (95% CI)",
        "figs/accepted_slo_miss_vs_load_ci.png",
    )
    _plot_metric_with_ci(
        stats_df,
        "accuracy_weighted_throughput_mean",
        "accuracy_weighted_throughput_ci95",
        "Accuracy-Weighted Throughput",
        "Accuracy-Weighted Throughput vs. Load (95% CI)",
        "figs/acc_throughput_vs_load_ci.png",
    )
    _plot_metric_with_ci(
        stats_df,
        "p99_latency_ms_mean",
        "p99_latency_ms_ci95",
        "P99 Latency of Completed Requests (ms)",
        "P99 Latency vs. Load (95% CI)",
        "figs/p99_latency_vs_load_ci.png",
    )

    print("Experiments completed. Results saved to 'results/' and 'figs/'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    run_experiment_sweep(args.config)
