import json
import os
import tempfile
import unittest

import yaml

from compute_intensive.metrics import build_failed_record, build_result_record, save_result
from compute_intensive.method_registry import (
    ALGORITHM_METHODS,
    ALL_METHODS,
    OFFLOADING_METHODS,
    get_methods_for_group,
    get_train_method,
)
from compute_intensive.plot_helpers import (
    DEFAULT_COMPUTE_SWEEP_TASK_RATE,
    is_compute_sweep_record,
    is_task_rate_sweep_record,
    load_training_rewards,
)


class MethodRegistryTests(unittest.TestCase):
    def test_offloading_group(self):
        names = [m.method_name for m in get_methods_for_group("offloading")]
        self.assertEqual(names, ["GO", "SSO", "MSCO"])

    def test_algorithm_group(self):
        names = [m.method_name for m in get_methods_for_group("algorithm")]
        self.assertEqual(names, ["GA", "DQN", "DGR-D3QN"])

    def test_ablation_group(self):
        names = [m.method_name for m in get_methods_for_group("ablation")]
        self.assertEqual(names, ["D3QN", "DGR-D3QN"])

    def test_train_methods(self):
        self.assertEqual(get_train_method("D3QN").config_file, "train_d3qn.yaml")

    def test_ga_raw_mode(self):
        ga = ALL_METHODS["GA"]
        self.assertEqual(ga.method_name, "GA")
        self.assertEqual(ga.raw_mode_name, "ICM")

    def test_go_uses_dgr_d3qn(self):
        go = OFFLOADING_METHODS["GO"]
        self.assertEqual(go.method_name, "GO")
        self.assertEqual(go.raw_mode_name, "DGR_D3QN")
        self.assertEqual(go.scheme, "GO")
        self.assertEqual(go.algorithm, "DGR-D3QN")

    def test_offloading_schemes(self):
        for name, scheme in (("GO", "GO"), ("SSO", "SSO"), ("MSCO", "MSCO")):
            spec = OFFLOADING_METHODS[name]
            self.assertEqual(spec.scheme, scheme)
            self.assertEqual(spec.algorithm, "DGR-D3QN")

    def test_algorithm_group_msco_scheme(self):
        for name in ("GA", "DQN", "DGR-D3QN"):
            self.assertEqual(ALGORITHM_METHODS[name].scheme, "MSCO")

    def test_ablation_group_msco_scheme(self):
        from compute_intensive.method_registry import ABLATION_METHODS

        for name, algorithm in (("D3QN", "D3QN"), ("DGR-D3QN", "DGR-D3QN")):
            spec = ABLATION_METHODS[name]
            self.assertEqual(spec.scheme, "MSCO")
            self.assertEqual(spec.algorithm, algorithm)


class MetricsTests(unittest.TestCase):
    def test_save_result_json_with_task_rate(self):
        record = build_result_record(
            method_name="SSO",
            scheme="SSO",
            algorithm="DGR-D3QN",
            raw_mode_name="DGR_D3QN",
            experiment_group="offloading",
            load_level=40,
            seed=42,
            statics_datas={
                "Total": 400,
                "Reached_0": 320,
                "Reached_1": 0,
                "Lost_upload": 10,
                "Lost_relay_0": 70,
                "Lost_relay_1": 0,
                "Total_delay_0": 3200.0,
                "Total_delay_1": 0,
                "Total_hops_0": 1600,
                "Total_hops_1": 0,
            },
            hop_count_list=[5.0],
            episode_reward_list=[],
            sweep_type="task_rate",
            task_rate_target=40,
            actual_task_rate=39.5,
            computing_ability=80e9,
            computing_ability_gflops=80,
            random_edges_del=15,
            removed_edges_count=15,
        )
        self.assertEqual(record["scheme"], "SSO")
        self.assertEqual(record["scenario"], "compute_intensive")
        self.assertEqual(record["algorithm"], "DGR-D3QN")
        self.assertEqual(record["sweep_type"], "task_rate")
        self.assertEqual(record["load_level"], 40)
        self.assertIsNone(record["failure_rate"])
        self.assertIsNone(record["average_episode_reward"])
        with tempfile.TemporaryDirectory() as tmp:
            path = save_result(record, tmp)
            self.assertIn("rate40", os.path.basename(path))
            with open(path, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            self.assertAlmostEqual(loaded["actual_task_rate"], 39.5)

    def test_failed_record_includes_computing_ability(self):
        record = build_failed_record(
            method_name="MSCO",
            scheme="MSCO",
            algorithm="DGR-D3QN",
            raw_mode_name="DGR_D3QN",
            experiment_group="offloading",
            load_level=60,
            seed=42,
            error_message="missing weights",
            sweep_type="compute",
            task_rate_target=60,
            computing_ability=80e9,
            computing_ability_gflops=80,
            random_edges_del=15,
        )
        self.assertEqual(record["computing_ability"], 80e9)
        self.assertEqual(record["computing_ability_gflops"], 80)
        self.assertEqual(record["random_edges_del"], 15)
        self.assertIsNone(record["failure_rate"])


class ConfigFileTests(unittest.TestCase):
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _load(self, name):
        path = os.path.join(self.ROOT, "configs", "compute_intensive", name)
        with open(path, "r", encoding="utf-8") as fp:
            return yaml.load(fp, Loader=yaml.FullLoader)

    def test_algo_dqn_uses_auto_mode(self):
        cfg = self._load("algo_dqn.yaml")
        agent = cfg["agent"]
        self.assertEqual(agent["mode"], "DQN")
        self.assertEqual(cfg["general"]["phase"], "test")

    def test_algo_d3qn_ablation(self):
        cfg = self._load("algo_d3qn.yaml")
        self.assertEqual(cfg["agent"]["mode"], "D3QN")
        self.assertEqual(cfg["experiment"]["experiment_group"], "ablation")

    def test_train_configs_share_task_rate(self):
        rates = [
            self._load(name)["environment"]["target_task_rate_per_sec"]
            for name in ("train_dqn.yaml", "train_d3qn.yaml", "train_dgr_d3qn.yaml")
        ]
        self.assertEqual(rates, [40, 40, 40])

    def test_offloading_configs_unified_dgr_d3qn(self):
        for name in ("offloading_go.yaml", "offloading_sso.yaml", "offloading_msco.yaml"):
            cfg = self._load(name)
            self.assertEqual(cfg["agent"]["mode"], "DGR_D3QN")
            self.assertEqual(cfg["experiment"]["algorithm"], "DGR-D3QN")
            self.assertEqual(
                cfg["agent"]["model_path"],
                "./model_weights/compute_intensive/train_dgr_d3qn.pth",
            )

    def test_algo_configs_msco_scheme(self):
        for name in ("algo_ga.yaml", "algo_dqn.yaml", "algo_dgr_d3qn.yaml"):
            cfg = self._load(name)
            self.assertEqual(cfg["experiment"]["scheme"], "MSCO")

    def test_offloading_configs_scenario(self):
        cfg = self._load("offloading_go.yaml")
        self.assertEqual(cfg["scenario"], "compute_intensive")
        self.assertEqual(cfg["environment"]["task_profile"], "compute_intensive")

    def test_random_edges_del_preserved(self):
        cfg = self._load("offloading_go.yaml")
        self.assertEqual(cfg["environment"]["random_edges_del"], 15)


class FlatD3QNTests(unittest.TestCase):
    def test_flat_d3qn_forward_with_mask(self):
        import torch
        from models.d3qn import FlatD3QNNetwork
        from models.state_layout import compute_dgr_packed_dim

        dim = compute_dgr_packed_dim()
        net = FlatD3QNNetwork(state_dim=dim)
        state = torch.randn(2, dim)
        mask = torch.tensor([[1, 1, 0, 0, 1], [0, 0, 0, 0, 1]], dtype=torch.float32)
        q = net(state, mask)
        self.assertEqual(q.shape, (2, 5))
        self.assertTrue(torch.isinf(q[0, 2]))


class PlotFilterTests(unittest.TestCase):
    def test_sweep_type_filters(self):
        task_rec = {"sweep_type": "task_rate", "computing_ability_gflops": 80}
        compute_rec = {
            "sweep_type": "compute",
            "task_rate_target": DEFAULT_COMPUTE_SWEEP_TASK_RATE,
            "computing_ability_gflops": 80,
            "experiment_group": "offloading",
        }
        self.assertTrue(is_task_rate_sweep_record(task_rec))
        self.assertFalse(is_compute_sweep_record(task_rec))
        self.assertTrue(is_compute_sweep_record(compute_rec))
        self.assertFalse(is_task_rate_sweep_record(compute_rec))

    def test_parse_training_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train_dqn.txt")
            with open(path, "w", encoding="utf-8") as fp:
                fp.write("Average ending reward: 0.5\n")
                fp.write("Average ending reward: None\n")
                fp.write("Average ending reward: 1.25\n")
            rewards = load_training_rewards(tmp, "DQN")
            self.assertEqual(rewards, [0.5, 1.25])

    def test_parse_training_jsonl_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train_dgr_d3qn.txt")
            with open(path, "w", encoding="utf-8") as fp:
                fp.write('{"episode_reward": 0.4, "moving_avg_reward_50": 0.35}\n')
                fp.write('{"episode_reward": 0.8, "moving_avg_reward_50": 0.55}\n')
            rewards = load_training_rewards(tmp, "DGR-D3QN")
            self.assertEqual(rewards, [0.4, 0.8])

    def test_parse_training_jsonl_latest_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train_dgr_d3qn.txt")
            with open(path, "w", encoding="utf-8") as fp:
                fp.write('{"run_id": "20250101_120000", "episode_reward": 0.1}\n')
                fp.write('{"run_id": "20250102_120000", "episode_reward": 0.4}\n')
                fp.write('{"run_id": "20250102_120000", "episode_reward": 0.8}\n')
            rewards = load_training_rewards(tmp, "DGR-D3QN")
            self.assertEqual(rewards, [0.4, 0.8])

    def test_training_log_relative_path(self):
        from compute_intensive.plot_helpers import training_log_relative_path

        self.assertTrue(
            training_log_relative_path("DGR-D3QN").endswith(
                os.path.join("training_process_data", "compute_intensive", "train_dgr_d3qn.txt")
            )
        )
        self.assertTrue(
            training_log_relative_path("DGR-D3QN", quick=True).endswith(
                os.path.join("training_process_data", "compute_intensive", "debug", "train_dgr_d3qn.txt")
            )
        )


if __name__ == "__main__":
    unittest.main()
