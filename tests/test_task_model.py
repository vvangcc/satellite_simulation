import random
import unittest

from task_model import (
    TaskInfo,
    attach_task_to_packet,
    build_compute_intensive_task,
    sync_packet_size,
)


class _MockPacket:
    def __init__(self, source, destination, creation_time, size):
        self.source = source
        self.destination = destination
        self.creation_time = creation_time
        self.size = size
        self.task = None


class Chapter3TaskGenerationTests(unittest.TestCase):
    def test_stage_count_in_range(self):
        rng = random.Random(0)
        seen = set()
        for _ in range(200):
            task = build_compute_intensive_task(
                task_type=0,
                destination="GS_1",
                birth_time=0.0,
                size_range=(25 * 1024 * 1024, 75 * 1024 * 1024),
                task_num_stages=[2, 3, 4],
                computing_demand_factor=(1400, 2200),
                size_reduction_factor=(1.3, 4.0),
                final_result_size_range=(5 * 1024, 15 * 1024),
                rng=rng,
            )
            seen.add(task.total_stages)
            self.assertIn(task.total_stages, (2, 3, 4))
            self.assertEqual(len(task.demand_seq_flops), task.total_stages)
            self.assertEqual(len(task.output_size_seq_bytes), task.total_stages)
            self.assertFalse(task.is_completed)
            self.assertEqual(task.stage_idx, 0)
            final_size = task.output_size_seq_bytes[-1]
            self.assertGreaterEqual(final_size, 5 * 1024)
            self.assertLessEqual(final_size, 15 * 1024)
        self.assertEqual(seen, {2, 3, 4})

    def test_demand_matches_input_size(self):
        rng = random.Random(1)
        task = build_compute_intensive_task(
            task_type=0,
            destination="GS_1",
            birth_time=1.0,
            size_range=(25 * 1024 * 1024, 75 * 1024 * 1024),
            task_num_stages=[3],
            computing_demand_factor=(1500, 1500),
            size_reduction_factor=(2.0, 2.0),
            final_result_size_range=(10 * 1024, 10 * 1024),
            rng=rng,
        )
        self.assertEqual(task.total_stages, 3)
        self.assertGreater(task.demand_seq_flops[0], 0)
        self.assertEqual(task.demand_seq_flops[0], int(1500 * task.current_size_bytes))


class MultiStageExecutionTests(unittest.TestCase):
    def _run_stages(self, total_stages: int):
        rng = random.Random(total_stages)
        task = build_compute_intensive_task(
            task_type=0,
            destination="GS_1",
            birth_time=0.0,
            size_range=(30 * 1024 * 1024, 30 * 1024 * 1024),
            task_num_stages=[total_stages],
            computing_demand_factor=(1400, 2200),
            size_reduction_factor=(1.3, 4.0),
            final_result_size_range=(5 * 1024, 15 * 1024),
            rng=rng,
        )
        packet = _MockPacket("SAT_1", task.destination, 0.0, task.current_size_bytes)
        attach_task_to_packet(packet, task)
        stage_trace = []

        for _ in range(total_stages):
            stage_idx_before = task.stage_idx
            demand = task.current_stage_demand
            old_size, new_size, completed_demand = task.complete_current_stage()
            sync_packet_size(packet)
            self.assertEqual(completed_demand, demand)
            stage_trace.append(
                {
                    "stage_idx_before": stage_idx_before,
                    "demand": completed_demand,
                    "old_size": old_size,
                    "new_size": new_size,
                    "is_completed": task.is_completed,
                }
            )

        self.assertTrue(task.is_completed)
        self.assertEqual(task.stage_idx, total_stages)
        self.assertEqual(len(stage_trace), total_stages)
        self.assertGreaterEqual(task.current_size_bytes, 5 * 1024)
        self.assertLessEqual(task.current_size_bytes, 15 * 1024)
        self.assertEqual(task.current_size_bytes, task.final_output_size)
        self.assertEqual(packet.size, task.final_output_size)
        for idx, record in enumerate(stage_trace):
            self.assertEqual(record["stage_idx_before"], idx)
            self.assertEqual(record["new_size"], task.output_size_seq_bytes[idx])
            if idx < total_stages - 1:
                self.assertFalse(record["is_completed"])
            else:
                self.assertTrue(record["is_completed"])
        return task

    def test_two_stage_task_executes_in_order(self):
        self._run_stages(2)

    def test_three_stage_task_executes_in_order(self):
        self._run_stages(3)

    def test_four_stage_task_executes_in_order(self):
        self._run_stages(4)

    def test_intermediate_stage_keeps_pipeline_open_until_last_stage(self):
        task = build_compute_intensive_task(
            task_type=0,
            destination="GS_1",
            birth_time=0.0,
            size_range=(30 * 1024 * 1024, 30 * 1024 * 1024),
            task_num_stages=[3],
            computing_demand_factor=(1400, 2200),
            size_reduction_factor=(1.3, 4.0),
            final_result_size_range=(5 * 1024, 15 * 1024),
            rng=random.Random(7),
        )
        task.complete_current_stage()
        self.assertFalse(task.is_completed)
        self.assertEqual(task.stage_idx, 1)
        self.assertGreater(task.current_stage_demand, 0)
        task.complete_current_stage()
        self.assertFalse(task.is_completed)
        task.complete_current_stage()
        self.assertTrue(task.is_completed)


class TaskInfoHelperTests(unittest.TestCase):
    def test_complete_current_stage_raises_when_already_done(self):
        task = TaskInfo(
            task_type=0,
            current_size_bytes=1024,
            demand_seq_flops=[100],
            output_size_seq_bytes=[512],
            total_stages=1,
            destination="GS_1",
        )
        task.complete_current_stage()
        with self.assertRaises(RuntimeError):
            task.complete_current_stage()


if __name__ == "__main__":
    unittest.main()
