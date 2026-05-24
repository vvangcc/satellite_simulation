from SatelliteNetworkSimulator_Beta import SatelliteNetworkSimulator, Packet, Satellite, Propagator
from task_model import (
    TaskInfo,
    attach_task_to_packet,
    build_compute_intensive_task,
    build_mixed_tasks_task,
    build_legacy_task,
    sync_packet_size,
)
from models.action_mask import build_action_mask, has_valid_actions, masked_argmax, random_valid_action
from policies.scheme_aware_dgr_d3qn import (
    bind_sso_compute_node,
    build_scheme_action_mask,
    is_go_scheme,
    resolve_decision_target,
    select_action_with_scheme,
)
from baselines.modes import is_heuristic_mode, allows_onboard_compute
from models.mixed_tasks_state_layout import (
    build_mixed_tasks_action_mask_from_layout,
    build_mixed_tasks_queue_state_vector,
    build_mixed_tasks_task_state_vector,
    is_mixed_tasks_profile,
    is_mixed_tasks_rl_mode,
    pack_mixed_tasks_state,
    task_is_completed_from_mixed_tasks_state,
    unpack_mixed_tasks_state,
    TASK_TYPE_DELAY_SENSITIVE,
)
from mixed_tasks.simulator_support import (
    build_mixed_tasks_reward,
    mixed_tasks_statics_defaults,
    d_hard_deadline_exceeded,
    init_mixed_tasks_queues,
    record_success,
    record_task_generated,
    within_c_soft_deadline,
    within_d_deadline,
)
from models.state_layout import (
    MAX_GRAPH_NODES,
    MAX_NEIGHBOR_SLOTS,
    TASK_IDX_IS_COMPLETED,
    build_action_mask_from_layout,
    build_task_state_vector,
    pack_dgr_state,
    task_is_completed_from_state,
    unpack_dgr_state,
)
import random
import numpy as np
import simpy
import networkx as nx
import torch

PRE = True
CT_FAC = 5
BITS_PER_BYTE = 8


def transmission_delay(size_bytes, rate_bps):
    """Compute link transmission delay; data size in bytes, rate in bps."""
    return size_bytes * BITS_PER_BYTE / rate_bps


def _loss_relay_stat(statics_data, task_type):
    if task_type == 0:
        statics_data['Lost_relay_0'] += 1
    else:
        statics_data['Lost_relay_1'] += 1


def _release_pre_reservation(graph, computing_node, demand):
    if computing_node and PRE and demand:
        graph.nodes[computing_node]['computing_remain'] -= demand


def _append_rl_experience(propagator, task, packet, satellite, reward, next_state, done):
    if task.last_state is None or task.last_action_mask is None:
        return
    next_action_mask = build_action_mask(packet, satellite, propagator.graph)
    propagator.experiences.append([
        task.last_state,
        task.last_action_mask,
        task.last_action,
        reward,
        next_state,
        next_action_mask,
        done,
    ])


def _is_rl_mode(mode: str) -> bool:
    return is_mixed_tasks_rl_mode(mode) or not is_heuristic_mode(mode)


def _uses_packed_dgr_state(mode: str) -> bool:
    """DGR-D3QN and flat D3QN share the same packed state / action-mask layout."""
    return mode in ("DGR_D3QN", "D3QN", "CH4_D3QN", "CH4_PER_D3QN")


def flush_rl_transition(propagator, satellite, packet, next_state):
    """Close the previous (s, a, r, s') tuple when a new decision state is reached."""
    task = packet.task
    if not _is_rl_mode(satellite.mode):
        return
    if task.last_state is None or task.last_action_mask is None or task.pending_reward is None:
        return
    _append_rl_experience(
        propagator, task, packet, satellite, task.pending_reward, next_state, task.pending_done
    )
    task.clear_pending_transition()


def finalize_rl_transition(propagator, satellite, packet, next_state, reward, done):
    """Terminal transition (downstream / drop) after the last action."""
    task = packet.task
    if not _is_rl_mode(satellite.mode):
        return
    if task.last_state is None or task.last_action_mask is None:
        return
    _append_rl_experience(propagator, task, packet, satellite, reward, next_state, done)
    task.clear_pending_transition()

class Reward_Function:
    def __init__(self, reach_factor, delay_factor, loss_factor, memory_threshold, memory_factor):
        self.reach_factor = reach_factor
        self.delay_factor = delay_factor
        self.loss_factor = loss_factor
        self.memory_threshold = memory_threshold
        self.memory_factor = memory_factor

    def reach_reward(self, delay):
        return self.reach_factor - self.delay_factor * delay

    def reach_reward_abnormal(self, delay):
        return -self.delay_factor * delay

    def normal_reward(self, delay, memory_remain):
        return -self.delay_factor * delay - self.memory_factor * (memory_remain < self.memory_threshold)

    def loss_reward(self, delay):
        return -self.loss_factor - self.delay_factor * delay


class Propagator_Computing(Propagator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiences = []
        self.final_rewards = []

    def trans_parameters(self, max_hop, downstream_delays, reward_function):
        self.max_hop = max_hop
        self.downstream_delays = downstream_delays
        self.reward_function = reward_function

    def reset_parameters(self):
        self.experiences = []
        self.final_rewards = []

    def propagate(self, node, next_hop, packet):
        task = packet.task
        if (node, next_hop) in self.propagation_delays:
            yield self.env.timeout(self.propagation_delays[(node, next_hop)])
            if next_hop in self.node_names:
                success = self.satellites[next_hop].push_forward(packet)
                if success:
                    self.logger.log(
                        f"Time {self.env.now:.3f}: {next_hop}: Packet {(packet.source, packet.destination, packet.creation_time)} received by router. Memory remain: {self.satellites[next_hop].current_memory_occupy}.",
                        detail=True)
                else:
                    source, destination, hops, creation_time, size = packet.source, packet.destination, packet.hops, packet.creation_time, packet.size
                    current_state = self.satellites[node].get_current_state(
                        destination, hops, task.is_completed, task,
                        self.satellites[node].max_size, self.satellites[node].computing_ability)
                    done = 1
                    sat_node = self.satellites[node]
                    reward = sat_node._reward_loss(task, self.env.now - creation_time)
                    finalize_rl_transition(
                        self, self.satellites[node], packet, current_state, reward, done)
                    self.final_rewards.append(reward)
                    _release_pre_reservation(self.graph, packet.computing_node, task.current_stage_demand)
                    _loss_relay_stat(self.statics_data, task.task_type)
                    self.logger.log(
                        f"Time {self.env.now:.3f}: {next_hop}: Routing queue is full, discarding packet {(packet.source, packet.destination, packet.creation_time)}.")
            else:
                _release_pre_reservation(self.graph, packet.computing_node, task.current_stage_demand)
                _loss_relay_stat(self.statics_data, task.task_type)
                self.logger.log(f"Time {self.env.now:.3f}: {next_hop} is missed, dropped 1 packet.")
        else:
            _release_pre_reservation(self.graph, packet.computing_node, task.current_stage_demand)
            _loss_relay_stat(self.statics_data, task.task_type)
            self.logger.log(f"Time {self.env.now:.3f}: connection {(node, next_hop)} is missed, dropped 1 packet")

    def downstream(self, node, packet):
        source, destination, hops, creation_time, size = packet.source, packet.destination, packet.hops, packet.creation_time, packet.size
        task = packet.task
        satellite = self.satellites[node]
        scheme = getattr(task, "scheme", None) or getattr(satellite, "scheme", None)
        ground_mode = is_go_scheme(scheme, satellite.mode)
        current_state = satellite.get_current_state(
            destination, hops, task.is_completed, task, satellite.max_size, satellite.computing_ability)
        done = 1
        if node in self.satellites:
            yield self.env.timeout(self.downstream_delays)
            success = ground_mode or task.is_completed
            delay = self.env.now - creation_time
            ch4 = is_mixed_tasks_profile(satellite.task_profile)
            if ch4 and success and task.is_completed:
                on_time = within_d_deadline(task, self.env.now) if task.task_type == TASK_TYPE_DELAY_SENSITIVE else within_c_soft_deadline(task, self.env.now)
                reward = self.reward_function.success_reward(task, delay, on_time)
                record_success(self.statics_data, task, self.env.now, creation_time)
                self.logger.log(
                    f"Time {self.env.now:.3f}: Packet {(source, destination, packet.creation_time)} reached destination {node} (mixed_tasks success).")
            elif ch4 and not success:
                reward = self.reward_function.failure_reward(task, delay)
                _release_pre_reservation(self.graph, packet.computing_node, task.current_stage_demand)
                _loss_relay_stat(self.statics_data, task.task_type)
                self.logger.log(
                    f"Time {self.env.now:.3f}: Packet {(source, destination, packet.creation_time)} downlinked before completion (mixed_tasks).")
            elif success:
                reward = self.reward_function.reach_reward(self.env.now - creation_time)
                if task.task_type == 0:
                    self.statics_data['Reached_after_computed_0'] += 1 if task.is_completed else 0
                    self.statics_data['Reached_0'] += 1
                    self.statics_data['Total_hops_0'] += packet.hops
                    self.statics_data['Total_delay_0'] += self.env.now - packet.creation_time
                else:
                    self.statics_data['Reached_after_computed_1'] += 1 if task.is_completed else 0
                    self.statics_data['Reached_1'] += 1
                    self.statics_data['Total_hops_1'] += packet.hops
                    self.statics_data['Total_delay_1'] += self.env.now - packet.creation_time
                if task.is_completed:
                    self.statics_data['Computing_waiting_time'] += packet.computing_waiting_time
                self.logger.log(
                    f"Time {self.env.now:.3f}: Packet {(source, destination, packet.creation_time)} reached destination {node} "
                    f"({'Ground offload' if ground_mode else 'after compute'}).")
            else:
                if ch4:
                    reward = self.reward_function.failure_reward(task, delay)
                else:
                    reward = satellite._reward_loss(task, delay)
                _release_pre_reservation(self.graph, packet.computing_node, task.current_stage_demand)
                _loss_relay_stat(self.statics_data, task.task_type)
                if not ch4:
                    self.logger.log(
                        f"Time {self.env.now:.3f}: Packet {(source, destination, packet.creation_time)} downlinked before all stages completed.")
        else:
            if ch4:
                reward = self.reward_function.failure_reward(task, self.env.now - creation_time)
            else:
                reward = self.reward_function.loss_reward(self.env.now - creation_time)
            _release_pre_reservation(self.graph, packet.computing_node, task.current_stage_demand)
            _loss_relay_stat(self.statics_data, task.task_type)
            self.logger.log(f"Time {self.env.now:.3f}: downlink of {node} is missed, dropped 1 packet")
        if task.last_state is not None:
            self.final_rewards.append(reward)
            finalize_rl_transition(self, satellite, packet, current_state, reward, done)


class Satellite_with_Computing(Satellite):
    def __init__(self,mode,select_mode,epsilon,max_hop,max_size,device,env,name,neighbors,memory,
                 computing_ability,transmission_rate,downlink_rate,state_update_period,is_downlink,logger,
                 statics_data={},num=None,processing_time=1e-9,heartbeat_timeout=0.25,task_profile='legacy',
                 scheme=None):
        self.num=num
        self.mode=mode
        self.scheme=scheme
        self.select_mode=select_mode
        self.q_net=None
        self.epsilon=epsilon
        self.max_hop=max_hop
        self.max_size=max_size
        self.device=device
        self.name=name
        self.neighbors= sorted(neighbors)
        self.env=env
        self.gat_node_dim=64
        self.orbit_altitude, self.orbit_number, self.sat_number = map(int, self.name.split('_')[1:])
        self.memory=memory
        self.computing_ability=computing_ability
        self.transmission_rate=transmission_rate
        self.downlink_rate=downlink_rate
        self.state_update_period=state_update_period
        self.logger=logger
        self.task_profile = task_profile
        self.computing_queue = simpy.Store(self.env)
        self.offload_queue = simpy.Store(self.env)
        self.offload_size = 0
        self.offload_length = 0
        self.transmission_queue = {neighbor: simpy.Store(self.env) for neighbor in self.neighbors}
        self.transmission_size ={neighbor: 0 for neighbor in self.neighbors}
        self.transmission_length ={neighbor: 0 for neighbor in self.neighbors}
        self.neighbor_hops = {neighbor: {} for neighbor in self.neighbors}
        self.current_queue_size=0
        self.current_computing_queue_size=0
        self.forward_queue = simpy.Store(self.env)
        self.current_memory_occupy=0
        self.active=True
        self.routing_tables={}
        if is_mixed_tasks_profile(task_profile):
            init_mixed_tasks_queues(self, env)
        if _uses_packed_dgr_state(self.mode):
            default_res = [1.0, 0.0, 0.0]
            self.neighbor_states = {neighbor: default_res[:] for neighbor in self.neighbors}
        elif 'New' in self.mode:
            self.current_state = [0,1,0,0,0,4,0,0,0,12,0,0]
            self.neighbor_states = {neighbor: [0,1,0,0,0,4,0,0,0,12,0,0] for neighbor in self.neighbors}
        else:
            self.current_state = [0,1,0,0]
            self.neighbor_states={neighbor: [0,1,0,0] for neighbor in self.neighbors}
        self.propagator=None
        self.statics_data=statics_data
        self.processing_time=processing_time
        self.heartbeat_timeout = heartbeat_timeout
        self.last_heartbeat = {neighbor: env.now for neighbor in self.neighbors}
        self.is_downlink=is_downlink
        self.is_producing=0

        self.hops={}
        self.adjacency_table = {self.name: (self.neighbors, self.env.now)}
        self.computing_remain=0
        self.neighbor_graph=None
        self.is_computing=False
        self.computing_time=0

        self.last_computing_time=0

    def trans_parameters(self,q_net, reward_function, direct_satellites):
        self.q_net=q_net
        self.reward_function = reward_function
        self.direct_satellites = direct_satellites

    def build_routing_table(self):
        result_dict = {}
        self.neighbor_hops = {neighbor: {} for neighbor in self.neighbors}
        for start in [self.name] + self.neighbors:
            queue = [(neighbor, [start, neighbor], 1) for neighbor in self.adjacency_table[start][0]]
            if start != self.name:
                self.neighbor_hops[start][start] = 0
            while queue:
                (node, path, hops) = queue.pop(0)
                if start == self.name:
                    if node not in result_dict:
                        result_dict[node] = ([path[1]], hops)
                        queue.extend((neighbor, path + [neighbor], hops + 1) for neighbor in self.adjacency_table[node][0] if neighbor not in path)
                    elif result_dict[node][1] == hops:
                        result_dict[node][0].append(path[1])
                else:
                    if node not in self.neighbor_hops[start]:
                        self.neighbor_hops[start][node] = hops
                        queue.extend((neighbor, path + [neighbor], hops + 1) for neighbor in self.adjacency_table[node][0] if neighbor not in path)
        result_dict[self.name] = ([self.name], 0)
        self.routing_tables = result_dict

    def push_forward(self,packet):
        if self.current_memory_occupy + packet.size < self.memory:
            self.current_memory_occupy += packet.size
            self.forward_queue.put(packet)
            if is_heuristic_mode(self.mode):
                if 'current_memory_occupy' in self.propagator.graph.nodes[self.name]:
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] += packet.size
                else:
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] = packet.size
            return True
        else:
            return False

    def push_computing(self,packet,computing_demand):
        self.current_computing_queue_size += packet.size
        self.computing_remain += computing_demand
        task = packet.task
        if is_mixed_tasks_profile(self.task_profile):
            if task.task_type == TASK_TYPE_DELAY_SENSITIVE:
                self.computing_queue_size_D += packet.size
                self.computing_queue_D.put(packet)
            else:
                self.computing_queue_size_C += packet.size
                self.computing_queue_C.put(packet)
        else:
            self.computing_queue.put(packet)
        if not PRE:
            if 'computing_remain' in self.propagator.graph.nodes[self.name]:
                self.propagator.graph.nodes[self.name]['computing_remain'] += computing_demand
            else:
                self.propagator.graph.nodes[self.name]['computing_remain'] = computing_demand

    def push_transmission(self,neighbor,packet):
        self.current_queue_size += packet.size
        self.transmission_length[neighbor]+=1
        self.transmission_size[neighbor]+=packet.size
        if is_mixed_tasks_profile(self.task_profile):
            task = packet.task
            if task.task_type == TASK_TYPE_DELAY_SENSITIVE:
                self.transmission_queue_size_D[neighbor] += packet.size
                self.transmission_queue_D[neighbor].put(packet)
            else:
                self.transmission_queue_size_C[neighbor] += packet.size
                self.transmission_queue_C[neighbor].put(packet)
        else:
            self.transmission_queue[neighbor].put(packet)
        if is_heuristic_mode(self.mode):
            if 'transmission_weight' in self.propagator.graph[self.name][neighbor]:
                self.propagator.graph[self.name][neighbor]['transmission_weight']+=packet.size
            else:
                self.propagator.graph[self.name][neighbor]['transmission_weight'] = packet.size

    def _reward_step(self, task, step_delay, memory_remain):
        if is_mixed_tasks_profile(self.task_profile):
            return self.reward_function.step_reward(task, step_delay, memory_remain)
        return self.reward_function.normal_reward(step_delay, memory_remain)

    def _reward_loss(self, task, total_delay):
        if is_mixed_tasks_profile(self.task_profile):
            return self.reward_function.loss_reward(task, total_delay)
        return self.reward_function.loss_reward(total_delay)

    def _get_mixed_tasks_compute_packet(self):
        while True:
            if self.computing_queue_D.items:
                packet = yield self.computing_queue_D.get()
                self.computing_queue_size_D = max(0, self.computing_queue_size_D - packet.size)
                return packet
            if self.computing_queue_C.items:
                packet = yield self.computing_queue_C.get()
                self.computing_queue_size_C = max(0, self.computing_queue_size_C - packet.size)
                return packet
            d_evt = self.computing_queue_D.get()
            c_evt = self.computing_queue_C.get()
            yield simpy.events.AnyOf(self.env, [d_evt, c_evt])

    def pop_transmission(self, neighbor):
        if is_mixed_tasks_profile(self.task_profile):
            while True:
                if self.transmission_queue_D[neighbor].items:
                    packet = yield self.transmission_queue_D[neighbor].get()
                    self.transmission_queue_size_D[neighbor] = max(
                        0, self.transmission_queue_size_D[neighbor] - packet.size
                    )
                    break
                elif self.transmission_queue_C[neighbor].items:
                    packet = yield self.transmission_queue_C[neighbor].get()
                    self.transmission_queue_size_C[neighbor] = max(
                        0, self.transmission_queue_size_C[neighbor] - packet.size
                    )
                    break
                else:
                    d_evt = self.transmission_queue_D[neighbor].get()
                    c_evt = self.transmission_queue_C[neighbor].get()
                    yield simpy.events.AnyOf(self.env, [d_evt, c_evt])
        else:
            packet = yield self.transmission_queue[neighbor].get()
        self.current_queue_size -= packet.size
        self.transmission_size[neighbor] -= packet.size
        self.transmission_length[neighbor] -= 1
        self.current_memory_occupy -= packet.size
        if is_heuristic_mode(self.mode):
            self.propagator.graph[self.name][neighbor]['transmission_weight'] -= packet.size
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size
        return packet

    def push_offload(self,packet):
        self.current_queue_size += packet.size
        self.offload_size+=packet.size
        self.offload_length+=1
        self.offload_queue.put(packet)

    def pop_offload(self):
        packet = yield self.offload_queue.get()
        self.current_queue_size-=packet.size
        self.offload_size-=packet.size
        self.offload_length-=1
        self.current_memory_occupy -= packet.size
        if is_heuristic_mode(self.mode):
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size
        return packet

    def get_resource_feature_vector(self):
        """h_i = [m_i, q_t_i, q_c_i], each normalized to [0, 1] scale."""
        m_i = 1.0 - min(self.current_memory_occupy / self.memory, 1.0)
        q_t_i = min(self.current_queue_size / self.memory, 1.0)
        remaining_compute = (
            self.computing_remain / self.computing_ability
            - self.is_computing * (self.env.now - self.last_computing_time)
        )
        q_c_i = min(max(remaining_compute / CT_FAC, 0.0), 1.0)
        return [m_i, q_t_i, q_c_i]

    def _neighbor_slots(self):
        slots = list(self.neighbors[:MAX_NEIGHBOR_SLOTS])
        while len(slots) < MAX_NEIGHBOR_SLOTS:
            slots.append(None)
        return slots

    def build_mixed_tasks_graph_state(self, destination, task_state):
        slots = self._neighbor_slots()
        node_feats = np.zeros((MAX_GRAPH_NODES, 3), dtype=np.float32)
        node_mask = np.zeros(MAX_GRAPH_NODES, dtype=np.float32)
        phys_adj = np.zeros((MAX_GRAPH_NODES, MAX_GRAPH_NODES), dtype=np.float32)
        node_feats[0] = self.get_resource_feature_vector()
        node_mask[0] = 1.0
        for slot_idx, neighbor in enumerate(slots):
            node_idx = slot_idx + 1
            if neighbor is not None and neighbor in self.neighbor_states:
                node_feats[node_idx] = np.asarray(self.neighbor_states[neighbor], dtype=np.float32)[:3]
                node_mask[node_idx] = 1.0
                phys_adj[0, node_idx] = 1.0
                phys_adj[node_idx, 0] = 1.0
        dest_state = np.full(MAX_NEIGHBOR_SLOTS, 2.0, dtype=np.float32)
        for slot_idx, neighbor in enumerate(slots):
            if neighbor is None:
                continue
            if destination in self.neighbor_hops.get(neighbor, {}):
                dest_state[slot_idx] = self.neighbor_hops[neighbor][destination] / self.max_hop
            elif destination in self.routing_tables:
                dest_state[slot_idx] = self.routing_tables[destination][1] / self.max_hop
        queue_state = build_mixed_tasks_queue_state_vector(self)
        return pack_mixed_tasks_state(task_state, queue_state, dest_state, node_feats, phys_adj, node_mask)

    def build_dgr_graph_state(self, destination, task_state):
        slots = self._neighbor_slots()
        node_feats = np.zeros((MAX_GRAPH_NODES, 3), dtype=np.float32)
        node_mask = np.zeros(MAX_GRAPH_NODES, dtype=np.float32)
        phys_adj = np.zeros((MAX_GRAPH_NODES, MAX_GRAPH_NODES), dtype=np.float32)

        node_feats[0] = self.get_resource_feature_vector()
        node_mask[0] = 1.0
        for slot_idx, neighbor in enumerate(slots):
            node_idx = slot_idx + 1
            if neighbor is not None and neighbor in self.neighbor_states:
                node_feats[node_idx] = np.asarray(self.neighbor_states[neighbor], dtype=np.float32)[:3]
                node_mask[node_idx] = 1.0
                phys_adj[0, node_idx] = 1.0
                phys_adj[node_idx, 0] = 1.0

        dest_state = np.full(MAX_NEIGHBOR_SLOTS, 2.0, dtype=np.float32)
        for slot_idx, neighbor in enumerate(slots):
            if neighbor is None:
                continue
            if destination in self.neighbor_hops.get(neighbor, {}):
                dest_state[slot_idx] = self.neighbor_hops[neighbor][destination] / self.max_hop
            elif destination in self.routing_tables:
                dest_state[slot_idx] = self.routing_tables[destination][1] / self.max_hop

        return pack_dgr_state(task_state, dest_state, node_feats, phys_adj, node_mask)

    def get_current_state(self, destination, hops, is_completed, task, max_size=None, computing_ability=None):
        max_size = max_size or self.max_size
        computing_ability = computing_ability or self.computing_ability
        if is_mixed_tasks_profile(self.task_profile):
            task_state = build_mixed_tasks_task_state_vector(
                task, max_size, computing_ability, self.max_hop, hops, self, destination
            )
            if is_mixed_tasks_rl_mode(self.mode) and self.mode != 'CH4_PPO':
                return self.build_mixed_tasks_graph_state(destination, task_state)
            current_state = []
            neighbors_state = []
            current_node_state = [
                self.is_producing,
                1 - self.current_memory_occupy / self.memory,
                (self.computing_remain / self.computing_ability - self.is_computing * (self.env.now - self.last_computing_time)) / CT_FAC,
            ]
            for neighbor in self.neighbors:
                neighbors_state.extend(self.neighbor_states[neighbor])
                neighbors_state.append(self.transmission_size[neighbor] / self.memory)
                if destination in self.neighbor_hops[neighbor]:
                    neighbors_state.append(self.neighbor_hops[neighbor][destination] / self.max_hop)
                else:
                    neighbors_state.append(2)
            if len(self.neighbors) < 4:
                for _ in range(4 - len(self.neighbors)):
                    neighbors_state.extend([1, 0, 1, 1, 1, 2])
            current_state.extend(neighbors_state)
            current_state.extend(current_node_state)
            current_state.extend(task_state.tolist())
            current_state.extend(build_mixed_tasks_queue_state_vector(self).tolist())
            return np.array(current_state, dtype=np.float32)
        task_state = build_task_state_vector(task, max_size, computing_ability, self.max_hop, hops)
        if _uses_packed_dgr_state(self.mode):
            return self.build_dgr_graph_state(destination, task_state)
        current_state = []
        neighbors_state = []
        current_node_state = [
            self.is_producing,
            1 - self.current_memory_occupy / self.memory,
            (self.computing_remain / self.computing_ability - self.is_computing * (self.env.now - self.last_computing_time)) / CT_FAC,
        ]
        for neighbor in self.neighbors:
            if 'New' in self.mode:
                neighbors_state.extend(
                    self.neighbor_states[neighbor][0:4]
                    + [x / 4 for x in self.neighbor_states[neighbor][4:8]]
                    + [x / 12 for x in self.neighbor_states[neighbor][8:12]]
                )
            else:
                neighbors_state.extend(self.neighbor_states[neighbor])
            neighbors_state.append(self.transmission_size[neighbor] / self.memory)
            if destination in self.neighbor_hops[neighbor]:
                neighbors_state.append(self.neighbor_hops[neighbor][destination] / self.max_hop)
            else:
                neighbors_state.append(2)
        if len(self.neighbors) < 4:
            if 'New' in self.mode:
                if neighbors_state:
                    av1, av2, av3 = (
                        2 * sum(neighbors_state[3::14]) / len(self.neighbors),
                        2 * sum(neighbors_state[7::14]) / len(self.neighbors),
                        2 * sum(neighbors_state[11::14]) / len(self.neighbors),
                    )
                else:
                    av1, av2, av3 = 1, 1, 1
            for _ in range(4 - len(self.neighbors)):
                if 'New' in self.mode:
                    neighbors_state.extend([1, 0, 1, av1, 1, 0, 1, av2, 1, 0, 1, av3, 1, 2])
                else:
                    neighbors_state.extend([1, 0, 1, 1, 1, 2])
        current_state.extend(neighbors_state)
        current_state.extend(current_node_state)
        current_state.extend(task_state.tolist())
        return np.array(current_state, dtype=np.float32)

    def _ensure_neighbor_graph(self):
        if self.neighbor_graph is None and self.propagator is not None:
            self.neighbor_graph = self.propagator.graph

    def _refresh_routing_weights(self):
        self._ensure_neighbor_graph()
        for edge in self.neighbor_graph.edges():
            node1, node2 = edge
            self.neighbor_graph[node1][node2]['weight'] = (
                self.neighbor_graph[node1][node2]['missing'] * 10
                + self.neighbor_graph[node1][node2].get('transmission_weight', 0) / self.transmission_rate
                + self.neighbor_graph[node1][node2]['propagation_weight']
            )
            if self.propagator.graph.nodes[node2].get('current_memory_occupy', 0) / self.memory > 0.9:
                self.neighbor_graph[node1][node2]['weight'] += 10

    def ground_routing(self, destination):
        """GO/Ground: Dijkstra shortest-delay path to downlink satellite, no onboard compute."""
        self._refresh_routing_weights()

        def shortest_path_and_cost(source, target):
            if source == target:
                return [], 0
            try:
                path = nx.shortest_path(self.neighbor_graph, source=source, target=target, weight='weight')
                if len(path) <= 1:
                    return [], 0
                next_hops = path[1:]
                cost = sum(self.neighbor_graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                return next_hops, cost
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return [], 10

        routing, _ = shortest_path_and_cost(self.name, destination)
        return routing, None

    def tradition_routing(self, current_state, computing=True):
        def shortest_path_and_cost(source, target):
            if source == target:
                return [], 0
            try:
                path = nx.shortest_path(self.neighbor_graph, source=source, target=target, weight='weight')
                if len(path) <= 1:
                    return [], 0
                next_hops = path[1:]
                cost = sum(self.neighbor_graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                return next_hops, cost
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return [], 10
        self._refresh_routing_weights()
        source, destination, size, computing_demand, size_after_computing, is_completed = current_state
        if is_completed:
            routing,_=shortest_path_and_cost(self.name, destination)
            return routing, None
        else:
            times={}
            routings={}
            n = max(int(len(self.neighbor_graph.nodes) * 1 / 2), 1)
            computing_nodes = [node for node in self.neighbor_graph.nodes if self.neighbor_graph.nodes[node].get('computing_remain', 0) == 0 and node!=destination]
            non_computing_nodes = sorted((node for node in self.neighbor_graph.nodes if self.neighbor_graph.nodes[node].get('computing_remain', 0) != 0 and node!=destination),key=lambda k: self.neighbor_graph.nodes[k].get('computing_remain', 0))
            computing_nodes += non_computing_nodes[:max(n - len(computing_nodes), 0)]
            for c_node in computing_nodes:
                time=self.neighbor_graph.nodes[c_node].get('computing_remain', 0)/self.computing_ability+ computing_demand/self.computing_ability + (self.propagator.graph.nodes[c_node].get('current_memory_occupy',0)/self.memory > 0.9)*10
                routing_1,time_1=shortest_path_and_cost(self.name, c_node)
                routing_2,time_2=shortest_path_and_cost(c_node, destination)
                times[c_node]=time+time_1+time_2
                times[c_node]+= transmission_delay(len(routing_1)*size+len(routing_2)*size_after_computing, self.transmission_rate)
                routings[c_node]=routing_1+routing_2
            if times:
                computing_decision=sorted(times, key=times.get)[0]
            else:
                return [None],None
            if PRE and computing:
                if 'computing_remain' in self.propagator.graph.nodes[computing_decision]:
                    self.propagator.graph.nodes[computing_decision]['computing_remain'] += computing_demand
                else:
                    self.propagator.graph.nodes[computing_decision]['computing_remain'] = computing_demand
            if not computing:
                return routings[computing_decision], None
            else:
                return routings[computing_decision],computing_decision

    def _heuristic_action_index(self, packet, destination, task):
        """Map heuristic policy to action index: 0..3 forward, 4 compute, 5 illegal."""
        if self.mode == "SSO" and not task.is_completed and task.fixed_compute_node == self.name:
            return 4
        if packet.computing_node == self.name and not task.is_completed:
            return 4
        if packet.routing:
            next_hop = packet.routing.pop(0)
            if next_hop is None:
                return 5
            if next_hop == self.name:
                if not task.is_completed:
                    return 4
                return 5
            if next_hop in self.neighbors:
                return self.neighbors.index(next_hop)
            return 5
        if self.mode == "SSO" and not task.is_completed and task.fixed_compute_node == self.name:
            return 4
        return 5

    def _plan_heuristic_routing(self, packet, current_state_legacy, computing):
        """Plan forward/compute route for heuristic modes (SSO / GA / Ground)."""
        if self.mode == "SSO":
            return self.sso_routing(current_state_legacy, packet, computing=computing)
        return self.tradition_routing(current_state_legacy, computing=computing)

    def sso_routing(self, current_state, packet, computing=True):
        """Single-satellite offloading: bind all stages to one compute node (GA/ICM-style search)."""
        task = packet.task
        source, destination, size, computing_demand, size_after_computing, is_completed = current_state
        if is_completed:
            routing, _ = self.tradition_routing(current_state, computing=False)
            return routing, None
        if task.fixed_compute_node is None:
            total_state = [
                source,
                destination,
                size,
                task.remaining_demand,
                task.final_output_size,
                False,
            ]
            _, chosen = self.tradition_routing(total_state, computing=computing)
            task.fixed_compute_node = chosen
        compute_node = task.fixed_compute_node
        if compute_node is None:
            return [None], None
        if self.name == compute_node:
            return [], compute_node
        stage_state = [source, destination, size, computing_demand, size_after_computing, False]
        routing, _ = self.tradition_routing(stage_state, computing=False)
        if routing and routing[0] == compute_node:
            return routing[:1], compute_node
        try:
            path = nx.shortest_path(self.neighbor_graph, source=self.name, target=compute_node, weight="weight")
            return path[1:], compute_node
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [None], compute_node

    def get_next_hop(self, current_state, destination, packet, action_mask=None):
        if action_mask is None:
            action_mask = build_action_mask(packet, self, self.propagator.graph)
        if not has_valid_actions(action_mask):
            return 5

        if _uses_packed_dgr_state(self.mode):
            if np.random.rand() <= self.epsilon:
                valid_actions = np.where(action_mask > 0.5)[0]
                if len(valid_actions) == 0:
                    return 0
                if np.random.rand() <= 0.5:
                    return random_valid_action(action_mask)
                dest_dirs = []
                slots = self._neighbor_slots()
                for slot_idx, neighbor in enumerate(slots):
                    if neighbor is None:
                        dest_dirs.append(2.0)
                    elif destination in self.neighbor_hops.get(neighbor, {}):
                        dest_dirs.append(self.neighbor_hops[neighbor][destination] / self.max_hop)
                    else:
                        dest_dirs.append(2.0)
                min_value = min(dest_dirs)
                nearest = [idx for idx, value in enumerate(dest_dirs) if value == min_value and action_mask[idx] > 0.5]
                if nearest:
                    return int(np.random.choice(nearest))
                return random_valid_action(action_mask)
            state_tensor = torch.tensor(current_state, dtype=torch.float).unsqueeze(0).to(self.device)
            mask_tensor = torch.tensor(action_mask, dtype=torch.float).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor, mask_tensor)
            return masked_argmax(q_values[0].detach().cpu().numpy(), action_mask)

        state_tensor = torch.tensor(current_state, dtype=torch.float).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float).unsqueeze(0).to(self.device)

        if 'DQN' in self.mode:
            if np.random.rand() <= self.epsilon:
                if np.random.rand() <= 0.5:
                    return random_valid_action(action_mask)
                neighbor_distances = []
                for neighbor in self.neighbors:
                    if destination in self.neighbor_hops[neighbor]:
                        neighbor_distances.append(self.neighbor_hops[neighbor][destination] / self.max_hop)
                    else:
                        neighbor_distances.append(2)
                if len(self.neighbors) < 4:
                    neighbor_distances.extend([2] * (4 - len(self.neighbors)))
                min_value = min(neighbor_distances)
                nearest = [idx for idx, value in enumerate(neighbor_distances) if value == min_value and action_mask[idx] > 0.5]
                if nearest:
                    return int(np.random.choice(nearest))
                return random_valid_action(action_mask)
            main_output = self.q_net(state_tensor).detach().cpu().numpy()[0]
            return masked_argmax(main_output, action_mask)
        if 'SAC' in self.mode:
            logits = self.q_net(state_tensor)
            logits = logits.masked_fill(mask_tensor <= 0, float('-inf'))
            probs = torch.nn.functional.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return [action.item(), log_prob.item()]
        logits = self.q_net(state_tensor)
        logits = logits.masked_fill(mask_tensor <= 0, float('-inf'))
        probs = torch.nn.functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return [action.item(), dist.log_prob(action).item()]

    def find_highest_score(self, destinations, task, hops):
        highest_score= -2
        best_destinations=[]
        for destination in destinations:
            if destination in self.routing_tables:
                current_state = self.get_current_state(destination, hops, task.is_completed, task)
                score = self.cal_score(current_state, task.is_completed, packet=None, task=task)
                if score > highest_score:
                    highest_score = score
                    best_destinations = [destination]
                elif score == highest_score:
                    best_destinations.append(destination)
        return best_destinations

    def cal_score(self, current_state, is_completed, packet=None, task=None):
        if packet is not None:
            action_mask = build_action_mask(packet, self, self.propagator.graph)
        elif _uses_packed_dgr_state(self.mode):
            if self.mode in ('CH4_D3QN', 'CH4_PER_D3QN'):
                task_state, _, _, _, node_mask = unpack_mixed_tasks_state(current_state)
                is_done = task_is_completed_from_mixed_tasks_state(task_state)
                action_mask = build_mixed_tasks_action_mask_from_layout(node_mask, is_done)
            else:
                task_state, _, _, _, node_mask = unpack_dgr_state(current_state)
                is_done = task_is_completed_from_state(task_state)
                action_mask = build_action_mask_from_layout(node_mask, is_done)
        else:
            action_mask = np.zeros(5, dtype=np.float32)
            action_mask[:min(len(self.neighbors), 4)] = 1.0
            if not is_completed:
                action_mask[4] = 1.0
        if _uses_packed_dgr_state(self.mode):
            state_tensor = torch.tensor(current_state, dtype=torch.float).unsqueeze(0).to(self.device)
            mask_tensor = torch.tensor(action_mask, dtype=torch.float).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor, mask_tensor)
            valid = np.where(action_mask > 0.5)[0]
            if len(valid) == 0:
                return float('-inf')
            return float(np.max(q_values[0].detach().cpu().numpy()[valid]))
        current_state_tensor = torch.tensor(current_state, dtype=torch.float).unsqueeze(0).to(self.device)
        main_output = self.q_net(current_state_tensor).detach().cpu().numpy()[0]
        valid = np.where(action_mask > 0.5)[0]
        if len(valid) == 0:
            return float('-inf')
        return float(np.max(main_output[valid]))

    def forward_packet(self):
        while self.active:
            packet = yield self.forward_queue.get()
            packet.hops += 1
            task = packet.task
            source, destination, hops, creation_time, size = packet.source, packet.destination, packet.hops, packet.creation_time, packet.size
            scheme = task.scheme or self.scheme
            decision_target = resolve_decision_target(task, packet, scheme)
            current_state = None
            action_mask = None
            reward = None
            done = 0
            yield self.env.timeout(self.processing_time)
            if is_mixed_tasks_profile(self.task_profile) and d_hard_deadline_exceeded(task, self.env.now):
                done = 1
                reward = self._reward_loss(task, self.env.now - creation_time)
                self.propagator.final_rewards.append(reward)
                if _is_rl_mode(self.mode):
                    terminal_state = self.get_current_state(destination, hops, task.is_completed, task)
                    finalize_rl_transition(self.propagator, self, packet, terminal_state, reward, done)
                task.failed = True
                self.current_memory_occupy -= size
                _release_pre_reservation(self.propagator.graph, packet.computing_node, task.current_stage_demand)
                _loss_relay_stat(self.statics_data, task.task_type)
                self.logger.log(f"Time {self.env.now:.3f}: D-task hard deadline exceeded, dropped packet")
                continue
            if self.mode == "Ground":
                if not packet.routing and destination != self.name:
                    packet.routing, packet.computing_node = self.ground_routing(destination)
            elif is_heuristic_mode(self.mode):
                task_type, current_size, stage_demand, stage_output = task.mission_state_raw()
                current_state_legacy = [source, destination, current_size, stage_demand, stage_output, task.is_completed]
                if not packet.routing and destination != self.name:
                    if task.is_completed:
                        packet.routing, packet.computing_node = self._plan_heuristic_routing(
                            packet, current_state_legacy, computing=False
                        )
                    elif not task.is_completed:
                        packet.routing, packet.computing_node = self._plan_heuristic_routing(
                            packet, current_state_legacy, computing=allows_onboard_compute(self.mode)
                        )
            else:
                if self.select_mode == 3 and destination != self.name:
                    min_hops_destinations = self.find_min_hops_destinations(5)
                    hightest_score_destinations = self.find_highest_score(min_hops_destinations, task, hops)
                    if hightest_score_destinations:
                        destination = np.random.choice(hightest_score_destinations)
                    else:
                        destination = 'False'
                    packet.destination = destination
                    task.destination = destination
                current_state = self.get_current_state(
                    decision_target, hops, task.is_completed, task)
                flush_rl_transition(self.propagator, self, packet, current_state)
            if not self.active:
                _release_pre_reservation(self.propagator.graph, packet.computing_node, task.current_stage_demand)
                _loss_relay_stat(self.statics_data, task.task_type)
                self.logger.log(f"Time {self.env.now:.3f}: {self.name} is missed, dropped 1 packet")
                break
            if destination == self.name and (
                task.is_completed or is_go_scheme(scheme, self.mode)
            ):
                reward = None
                self.push_offload(packet)
            elif destination != self.name or not task.is_completed:
                if hops <= 2 * self.max_hop:
                    action_mask = None
                    if is_heuristic_mode(self.mode):
                        action = self._heuristic_action_index(packet, destination, task)
                    else:
                        if current_state is None:
                            current_state = self.get_current_state(
                                decision_target, hops, task.is_completed, task
                            )
                        if _uses_packed_dgr_state(self.mode):
                            action_mask = build_scheme_action_mask(
                                packet, self, self.propagator.graph, scheme
                            )
                        else:
                            action_mask = build_action_mask(packet, self, self.propagator.graph)
                        if not has_valid_actions(action_mask):
                            done = 1
                            reward = self._reward_loss(task, self.env.now - creation_time)
                            self.propagator.final_rewards.append(reward)
                            finalize_rl_transition(self.propagator, self, packet, current_state, reward, done)
                            self.current_memory_occupy -= size
                            _release_pre_reservation(self.propagator.graph, packet.computing_node, task.current_stage_demand)
                            _loss_relay_stat(self.statics_data, task.task_type)
                            self.logger.log(f"Time {self.env.now:.3f}: no legal action, dropped 1 packet")
                            continue
                        if _uses_packed_dgr_state(self.mode):
                            action = select_action_with_scheme(
                                packet,
                                self,
                                self.propagator.graph,
                                scheme,
                                self.q_net,
                                self.env,
                                current_state=current_state,
                                decision_target=decision_target,
                            )
                        else:
                            action = self.get_next_hop(current_state, destination, packet, action_mask)
                    if isinstance(action, (list, tuple)):
                        next_index = action[0]
                    else:
                        next_index = action
                    if destination in self.routing_tables:
                        if next_index < len(self.neighbors):
                            done = 0
                            reward = self._reward_step(
                                task, self.env.now - task.last_decision_time, 1 - self.current_memory_occupy / self.memory)
                            next_hop = self.neighbors[next_index]
                            if action_mask is not None:
                                task.record_decision(self.env.now, current_state, action, action_mask=action_mask)
                                task.set_pending_transition(reward, done)
                            else:
                                task.record_decision(self.env.now, current_state, action)
                            self.push_transmission(next_hop, packet)
                        elif next_index == 4 and not task.is_completed:
                            done = 0
                            packet.hops -= 1
                            bind_sso_compute_node(task, self.name, task.scheme or self.scheme)
                            reward = self._reward_step(
                                task, self.env.now - task.last_decision_time, 1 - self.current_memory_occupy / self.memory)
                            if action_mask is not None:
                                task.record_decision(self.env.now, current_state, action, action_mask=action_mask)
                                task.set_pending_transition(reward, done)
                            else:
                                task.record_decision(self.env.now, current_state, action)
                            packet.computing_node = None
                            self.push_computing(packet, task.current_stage_demand)
                        else:
                            done = 1
                            reward = self._reward_loss(task, self.env.now - creation_time)
                            self.propagator.final_rewards.append(reward)
                            if action_mask is not None:
                                finalize_rl_transition(self.propagator, self, packet, current_state, reward, done)
                            self.current_memory_occupy -= size
                            _release_pre_reservation(self.propagator.graph, packet.computing_node, task.current_stage_demand)
                            _loss_relay_stat(self.statics_data, task.task_type)
                            self.logger.log(f"Time {self.env.now:.3f}: wrong forward decision, dropped 1 packet")
                    else:
                        self.current_memory_occupy -= size
                        _release_pre_reservation(self.propagator.graph, packet.computing_node, task.current_stage_demand)
                        _loss_relay_stat(self.statics_data, task.task_type)
                        self.logger.log(f"Time {self.env.now:.3f}: {destination} is missed, dropped 1 packet")
                        reward = None
                else:
                    self.current_memory_occupy -= size
                    done = 1
                    reward = self._reward_loss(task, self.env.now - creation_time)
                    self.propagator.final_rewards.append(reward)
                    if _is_rl_mode(self.mode):
                        terminal_state = self.get_current_state(destination, hops, task.is_completed, task)
                        finalize_rl_transition(self.propagator, self, packet, terminal_state, reward, done)
                    _release_pre_reservation(self.propagator.graph, packet.computing_node, task.current_stage_demand)
                    _loss_relay_stat(self.statics_data, task.task_type)
                    self.logger.log(f"Time {self.env.now:.3f}: transmission out of time, dropped 1 packet")
            if task.last_state is not None and reward is not None and _is_rl_mode(self.mode):
                pass  # transitions are flushed at the next decision point or terminal handler

    def transmit_packet(self,neighbor):
        while self.active:
            packet = yield self.env.process(self.pop_transmission(neighbor))
            task = packet.task
            tx_wait = self.env.now - task.last_decision_time
            task.transmission_queue_time_sec += tx_wait
            tx_time = transmission_delay(packet.size, self.transmission_rate)
            yield self.env.timeout(tx_time)
            task.transmission_time_sec += tx_time
            if neighbor not in self.neighbors or not self.active:
                _release_pre_reservation(self.propagator.graph, packet.computing_node, task.current_stage_demand)
                _loss_relay_stat(self.statics_data, task.task_type)
                self.logger.log(f"Time {self.env.now:.3f}: transmission stopped, dropped 1 packet")
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination,packet.creation_time)} departed. Memory remain: {(self.memory-self.current_memory_occupy)}",detail=True)
                self.env.process(self.propagator.propagate(self.name,neighbor, packet))

    def offload_to_ground(self):
        while self.active:
            packet = yield self.env.process(self.pop_offload())
            task = packet.task
            tx_wait = self.env.now - task.last_decision_time
            task.transmission_queue_time_sec += tx_wait
            tx_time = transmission_delay(packet.size, self.downlink_rate)
            yield self.env.timeout(tx_time)
            task.transmission_time_sec += tx_time
            if not self.active:
                _release_pre_reservation(self.propagator.graph, packet.computing_node, task.current_stage_demand)
                _loss_relay_stat(self.statics_data, task.task_type)
                self.logger.log(f"Time {self.env.now:.3f}: offload stopped, dropped 1 packet")
                break
            self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination,packet.creation_time)} is offloading. Memory remain: {(self.memory-self.current_memory_occupy)}",detail=True)
            self.env.process(self.propagator.downstream(self.name, packet))

    def computing_packet(self):
        while self.active:
            if is_mixed_tasks_profile(self.task_profile):
                packet = yield self.env.process(self._get_mixed_tasks_compute_packet())
            else:
                packet = yield self.computing_queue.get()
            task = packet.task
            self.is_computing = True
            self.last_computing_time = self.env.now
            queue_wait = self.env.now - task.last_decision_time
            task.computing_queue_time_sec += queue_wait
            packet.computing_waiting_time += queue_wait
            stage_demand = task.current_stage_demand
            computing_time_consume = stage_demand / self.computing_ability
            yield self.env.timeout(computing_time_consume)
            task.computing_time_sec += computing_time_consume
            self.computing_time += computing_time_consume
            self.computing_remain -= stage_demand
            if is_heuristic_mode(self.mode):
                self.propagator.graph.nodes[self.name]['computing_remain'] -= stage_demand
            old_size, new_size, _ = task.complete_current_stage()
            if is_heuristic_mode(self.mode):
                self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= old_size - new_size
            self.current_computing_queue_size -= old_size
            self.current_memory_occupy -= old_size - new_size
            sync_packet_size(packet)
            self.is_computing = False
            self.last_computing_time = 0
            if not self.active:
                self.current_memory_occupy -= packet.size
                if is_heuristic_mode(self.mode):
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size
                _release_pre_reservation(self.propagator.graph, packet.computing_node, task.current_stage_demand)
                _loss_relay_stat(self.statics_data, task.task_type)
                self.logger.log(f"Time {self.env.now:.3f}: transmission stopped, dropped 1 packet")
                break
            task.record_decision(self.env.now)
            stage_label = f"stage {task.stage_idx}/{task.total_stages}" if not task.is_completed else "all stages completed"
            self.logger.log(
                f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source, packet.destination, packet.creation_time)} finished {stage_label}. Memory remain: {(self.memory - self.current_memory_occupy)}",
                detail=True)
            self.forward_queue.put(packet)

    def add_neighbor(self,neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.neighbors=sorted(self.neighbors)
            self.transmission_size[neighbor] = 0
            self.transmission_length[neighbor] = 0
            self.transmission_queue[neighbor] = simpy.Store(self.env)
            if is_mixed_tasks_profile(self.task_profile):
                self.transmission_queue_D[neighbor] = simpy.Store(self.env)
                self.transmission_queue_C[neighbor] = simpy.Store(self.env)
                self.transmission_queue_size_D[neighbor] = 0
                self.transmission_queue_size_C[neighbor] = 0
            if "New" in self.mode:
                self.neighbor_states[neighbor] = [0,1,0,0,0,4,0,0,0,12,0,0]
            elif _uses_packed_dgr_state(self.mode):
                self.neighbor_states[neighbor] = [1.0, 0.0, 0.0]
            else:
                self.neighbor_states[neighbor] = [0,1,0,0]
            self.last_heartbeat[neighbor] = self.env.now
            self.adjacency_table [self.name]=(self.neighbors, self.env.now)
            self.neighbor_hops[neighbor] = {}
            self.adjacency_table_exchanger()
            self.env.process(self.monitor_single_neighbor(neighbor))

    def del_neighbor(self,neighbor):
        if self.active:
            if neighbor in self.neighbors:
                while self.transmission_queue[neighbor].items:
                    packet = yield self.env.process(self.pop_transmission(neighbor))
                    task = packet.task
                    task.record_decision(self.env.now)
                    packet.routing = None
                    _release_pre_reservation(self.propagator.graph, packet.computing_node, task.current_stage_demand)
                    packet.computing_node = None
                    if not is_heuristic_mode(self.mode):
                        success = self.push_forward(packet)
                    else:
                        success = False
                    if not success:
                        _loss_relay_stat(self.statics_data, task.task_type)
                    self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
                if neighbor in self.neighbors:
                    self.neighbors.remove(neighbor)
                del self.transmission_size[neighbor]
                del self.neighbor_states[neighbor]
                del self.neighbor_hops[neighbor]
                del self.transmission_length[neighbor]
                self.adjacency_table[self.name]=(self.neighbors, self.env.now)
                self.update_adjacency_dict_for_bfs()
                self.build_routing_table()
                self.adjacency_table_exchanger()
                return True
            else:
                return False
        else:
            return False

    def find_min_hops_destinations(self, n):
        hop_counts = []
        for destination in self.direct_satellites:
            if destination in self.routing_tables:
                hops = self.routing_tables[destination][1]
            else:
                hops = np.inf
            hop_counts.append((destination, hops))
        if len(hop_counts)>n:
            hop_counts.sort(key=lambda x: x[1])
            return [a for a,b in hop_counts[:n]]
        else:
            return [a for a,b in hop_counts]

    def state_exchanger(self):
        while self.active:
            yield self.env.timeout(self.state_update_period)
            if not self.active:
                break
            for neighbor in self.neighbors:
                if _uses_packed_dgr_state(self.mode):
                    self.env.process(
                        self.propagator.send_state(self.name, neighbor, self.get_resource_feature_vector())
                    )
                elif 'New' in self.mode:
                    n = len(self.neighbors)
                    if n:
                        position_sums = [sum(items) for items in zip(*self.neighbor_states.values())]
                        av1, av2, av3 = 2 * position_sums[3] / n, 2 * position_sums[7] / n, 2 * position_sums[11] / n
                    else:
                        av1, av2, av3 = 1, 1, 1
                    specified_values_list = [1,0,1,av1,1,0,1,av2,1,0,1,av3]
                    self_value=[self.is_producing, 1 - self.current_memory_occupy / self.memory,(self.computing_remain / self.computing_ability-self.is_computing*(self.env.now-self.last_computing_time))/ CT_FAC ,sum(self.transmission_size.values())/self.memory]
                    self_values= [0,0,0,0]+[x * n for x in self.current_state]+[0,0,0,0]
                    additional_values = [x * (4 - n) for x in specified_values_list]
                    temp = [sum(tup) + add - sub for tup, add,sub in zip(zip(*self.neighbor_states.values()), additional_values,self_values)]
                    self.current_state=self_value+temp[0:8]
                    self.env.process(self.propagator.send_state(self.name, neighbor, self.current_state))
                else:
                    self.env.process(self.propagator.send_state(self.name, neighbor,[self.is_producing,1-self.current_memory_occupy/self.memory,(self.computing_remain / self.computing_ability-self.is_computing*(self.env.now-self.last_computing_time))/ CT_FAC,self.transmission_size[neighbor]/self.memory]))

    def all_start(self):
        super().all_start()
        self.env.process(self.computing_packet())
        self.env.process(self.offload_to_ground())

    def self_missing(self):
        self.active = False
        while self.computing_queue.items:
            packet=yield self.computing_queue.get()
            self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
        self.current_computing_queue_size = 0
        while self.offload_queue.items:
            packet = yield self.env.process(self.pop_offload())
            self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
        for neighbor in self.neighbors:
            while self.transmission_queue[neighbor].items:
                packet = yield self.env.process(self.pop_transmission(neighbor))
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
            while self.forward_queue.items:
                packet = yield self.env.process(self.forward_queue.get())
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
        self.current_memory_occupy=0
        if is_heuristic_mode(self.mode):
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] = 0
        self.is_producing = 0
        self.computing_remain = 0


class SatelliteNetworkSimulator_OnbardComputing(SatelliteNetworkSimulator):
    def __init__(self,mode,select_mode,q_net,epsilon,reward_factors,device,mission_possibility,poisson_rate,packet_frequency,computing_demand_factor,computing_demand_factor_2,size_after_computing_factor,size_after_computing_1,graph,landmarks,mean_interval_time,memory,computing_ability,transmission_rate,downlink_rate,downstream_delays,packet_size_range,state_update_period,logger,task_profile='legacy',task_num_stages=None,size_reduction_factor=None,final_result_size_range=None,heartbeat_timeout=0.25,target_task_rate_per_sec=None,scheme=None):
        self.q_net = q_net
        self.mode=mode
        self.scheme=scheme
        self.epsilon=epsilon
        if is_mixed_tasks_profile(task_profile):
            reward_cfg = reward_factors if isinstance(reward_factors, dict) else {}
            self.reward_function = build_mixed_tasks_reward(reward_cfg)
        else:
            self.reward_function = Reward_Function(*reward_factors)
        self.device=device
        self.mission_possibility=mission_possibility
        self.poisson_rate=poisson_rate
        self.packet_frequency=packet_frequency
        self.target_task_rate_per_sec = target_task_rate_per_sec
        self.use_global_task_generator = (
            target_task_rate_per_sec is not None and float(target_task_rate_per_sec) > 0
        )
        self.computing_demand_factor=computing_demand_factor
        self.computing_demand_factor_2=computing_demand_factor_2
        self.size_after_computing_factor=size_after_computing_factor
        self.size_after_computing_1=size_after_computing_1
        self.task_profile = task_profile
        self.task_num_stages = task_num_stages or [2, 3, 4]
        self.size_reduction_factor = size_reduction_factor or [1.3, 4.0]
        self.final_result_size_range = final_result_size_range or [5 * 1024, 15 * 1024]
        self.heartbeat_timeout = heartbeat_timeout
        self.graph = graph
        self.max_hop=nx.diameter(self.graph)
        self.max_size=packet_size_range[1]
        self.env = simpy.Environment()
        self.memory = memory
        self.computing_ability = computing_ability
        self.logger=logger
        self.transmission_rate=transmission_rate
        self.downstream_delays=downstream_delays
        self.downlink_rate=downlink_rate
        self.state_update_period=state_update_period
        self.landmarks=landmarks
        self.direct_satellites = set(sum(self.landmarks.values(), []))
        self.statics_datas = {
            'Total': 0, 'Reached_0': 0, 'Reached_1': 0, 'Reached_after_computed_0': 0,
            'Reached_after_computed_1': 0, 'Lost_upload': 0, 'Lost_relay_0': 0, 'Lost_relay_1': 0,
            'Total_delay_0': 0, 'Total_delay_1': 0, 'Total_hops_0': 0, 'Total_hops_1': 0,
            'Is_computing': 0, 'Computing_waiting_time': 0,
        }
        if is_mixed_tasks_profile(task_profile):
            self.statics_datas.update(mixed_tasks_statics_defaults())
        self.satellite_names=[node for node in self.graph.nodes]
        self.satellites={}
        self.select_mode=select_mode
        for node in self.graph.nodes:
            satellite_name_with_suffix = node
            self.satellites[satellite_name_with_suffix] = Satellite_with_Computing(
                self.mode,
                self.select_mode,
                self.epsilon,
                self.max_hop,
                self.max_size,
                self.device,
                self.env,
                satellite_name_with_suffix,
                [neighbor for neighbor in self.graph.neighbors(node)],
                self.memory,
                self.computing_ability,
                self.transmission_rate,
                self.downlink_rate,
                self.state_update_period,
                True if node in self.direct_satellites else False,
                self.logger,
                self.statics_datas,
                heartbeat_timeout=self.heartbeat_timeout,
                task_profile=self.task_profile,
                scheme=self.scheme,
            )
        self.propagator = Propagator_Computing(self.env, graph, logger, self.satellites,self.statics_datas, is_heuristic_mode(self.mode))
        self.mean_interval_time=mean_interval_time
        self.size_range = packet_size_range
        self.propagator.trans_parameters(self.max_hop,self.downstream_delays,self.reward_function)
        for satellite in self.satellites:
            self.satellites[satellite].adjacency_table=self.extract_adjacency_dict()
            self.satellites[satellite].trans_parameters(self.q_net,self.reward_function,self.direct_satellites)
            self.satellites[satellite].set_propagator(self.propagator)
            self.satellites[satellite].build_routing_table()

    def extract_adjacency_dict(self):
        adjacency_dict = {}
        for node in self.satellite_names:
            neighbors = [f"{neighbor}" for neighbor in self.graph.neighbors(node)]
            adjacency_dict[f"{node}"] = (neighbors, self.env.now)
        return adjacency_dict

    def _build_task(self, task_type, destination, birth_time):
        if self.task_profile == 'mixed_tasks':
            return build_mixed_tasks_task(
                task_type=task_type,
                destination=destination,
                birth_time=birth_time,
                size_range=self.size_range,
                task_num_stages=self.task_num_stages,
                computing_demand_factor=self.computing_demand_factor,
                size_reduction_factor=self.size_reduction_factor,
                final_result_size_range=self.final_result_size_range,
            )
        if self.task_profile == 'compute_intensive':
            return build_compute_intensive_task(
                task_type=task_type,
                destination=destination,
                birth_time=birth_time,
                size_range=self.size_range,
                task_num_stages=self.task_num_stages,
                computing_demand_factor=self.computing_demand_factor,
                size_reduction_factor=self.size_reduction_factor,
                final_result_size_range=self.final_result_size_range,
                scheme=self.scheme,
            )
        return build_legacy_task(
            task_type=task_type,
            destination=destination,
            birth_time=birth_time,
            size_range=self.size_range,
            computing_demand_factor=self.computing_demand_factor,
            computing_demand_factor_2=self.computing_demand_factor_2,
            size_after_computing_factor=self.size_after_computing_factor,
            size_after_computing_1=self.size_after_computing_1,
        )

    def _non_direct_satellite_names(self):
        return [s for s in self.satellite_names if s not in self.direct_satellites]

    def _pick_destination_for_source(self, satellite):
        min_hops = np.inf
        min_hops_destinations = []
        for destination in self.direct_satellites:
            if destination in self.satellites[satellite].routing_tables:
                hops = self.satellites[satellite].routing_tables[destination][1]
            else:
                hops = np.inf
            if hops < min_hops:
                min_hops = hops
                min_hops_destinations = [destination]
            elif hops == min_hops:
                min_hops_destinations.append(destination)
        if not min_hops_destinations:
            return None, None
        return np.random.choice(min_hops_destinations), min_hops

    def _update_heuristic_neighbor_graph(self, satellite, destination, min_hops):
        if not is_heuristic_mode(self.mode):
            return
        neighbors = []
        for _satellite in self.satellites:
            if (
                satellite in self.satellites[_satellite].routing_tables
                and destination in self.satellites[_satellite].routing_tables
                and (
                    self.satellites[_satellite].routing_tables[satellite][1]
                    + self.satellites[_satellite].routing_tables[destination][1]
                )
                <= min_hops + 4
            ):
                neighbors.append(self.satellites[_satellite].name)
        self.satellites[satellite].neighbor_graph = self.propagator.graph.subgraph(neighbors)

    def _spawn_packet(self, satellite, destination, task_type, min_hops=None):
        if satellite not in self.satellite_names or destination not in self.satellite_names:
            return False
        task = self._build_task(task_type, destination, self.env.now)
        if is_mixed_tasks_profile(self.task_profile):
            record_task_generated(self.statics_datas, task)
        if self.select_mode != 1:
            min_hops_destinations = self.satellites[satellite].find_min_hops_destinations(5)
            hightest_score_destinations = self.satellites[satellite].find_highest_score(
                min_hops_destinations, task, 0
            )
            if hightest_score_destinations:
                destination = np.random.choice(hightest_score_destinations)
                task.destination = destination
            else:
                return False
        if destination not in self.satellites[satellite].routing_tables:
            self.logger.log(
                f"Time {self.env.now:.3f}: {destination} is missed, packet failed to generate."
            )
            return False
        if min_hops is not None:
            self._update_heuristic_neighbor_graph(satellite, destination, min_hops)
        packet = Packet(satellite, destination, self.env.now, task.current_size_bytes)
        attach_task_to_packet(packet, task)
        self.statics_datas['Total'] += 1
        self.logger.log(
            f"Time {self.env.now:.3f}: {satellite}: Packet generated: {(satellite, destination, packet.creation_time)}."
        )
        success = self.satellites[satellite].push_forward(packet)
        if success:
            self.logger.log(
                f"Time {self.env.now:.3f}: {satellite}: Packet {(packet.source, packet.destination, packet.creation_time)} received by router. Memory remain: {self.satellites[satellite].current_memory_occupy}.",
                detail=True,
            )
        else:
            self.statics_datas['Lost_upload'] += 1
            self.logger.log(
                f"Time {self.env.now:.3f}: {satellite}: Routing queue is full, discarding packet ({satellite}, {destination}, packet.creation_time)."
            )
        return success

    def generate_global_traffic(self):
        """Network-wide Poisson task arrivals at target_task_rate_per_sec."""
        rate = float(self.target_task_rate_per_sec)
        while True:
            inter_arrival = random.expovariate(rate)
            yield self.env.timeout(inter_arrival)
            candidates = self._non_direct_satellite_names()
            if not candidates:
                continue
            satellite = random.choice(candidates)
            if satellite not in self.satellite_names:
                continue
            task_type = random.choices([0, 1], weights=self.mission_possibility, k=1)[0]
            min_hops = None
            if self.select_mode == 1 or is_heuristic_mode(self.mode):
                destination, min_hops = self._pick_destination_for_source(satellite)
                if destination is None:
                    continue
            else:
                destination, min_hops = self._pick_destination_for_source(satellite)
                if destination is None:
                    continue
            self.satellites[satellite].is_producing = 1
            self._spawn_packet(satellite, destination, task_type, min_hops)
            self.satellites[satellite].is_producing = 0

    def generate_traffic(self, satellite):
        while satellite in self.satellite_names:
            self.satellites[satellite].is_producing=0
            session_start_time = min(random.expovariate(1.0 / self.poisson_rate),self.poisson_rate*3)
            yield self.env.timeout(session_start_time)
            if not satellite in self.satellite_names:
                self.logger.log(f"Time {self.env.now:.3f}: {satellite} is missed, packets failed to generate.")
                break
            type = random.choices([0, 1], weights=self.mission_possibility, k=1)[0]
            if satellite in self.direct_satellites:
                continue
            else:
                if self.select_mode == 1 or is_heuristic_mode(self.mode):
                    min_hops = np.inf
                    min_hops_destinations=[]
                    for destination in self.direct_satellites:
                        if destination in self.satellites[satellite].routing_tables:
                            hops = self.satellites[satellite].routing_tables[destination][1]
                        else:
                            hops = np.inf
                        if hops < min_hops:
                            min_hops = hops
                            min_hops_destinations = [destination]
                        elif hops == min_hops:
                            min_hops_destinations.append(destination)
                    if min_hops_destinations:
                        destination = np.random.choice(min_hops_destinations)
                    else:
                        self.logger.log(f"Time {self.env.now:.3f}: connections for {satellite} is not availiable, packet failed to generate.")
            session_duration = min(random.expovariate(1.0 / self.mean_interval_time),self.mean_interval_time*3)

            end_time = self.env.now + session_duration
            self.satellites[satellite].is_producing = 1
            if is_heuristic_mode(self.mode):
                neighbors=[]
                for _satellite in self.satellites:
                    if satellite in self.satellites[_satellite].routing_tables and destination in self.satellites[_satellite].routing_tables:
                        if (self.satellites[_satellite].routing_tables[satellite][1]+self.satellites[_satellite].routing_tables[destination][1])<=min_hops+4:
                            neighbors.append(self.satellites[_satellite].name)
                self.satellites[satellite].neighbor_graph=self.propagator.graph.subgraph(neighbors)
            while self.env.now < end_time:
                yield self.env.timeout(1.0 / self.packet_frequency)
                self._spawn_packet(satellite, destination, type, min_hops)
                if destination not in self.satellite_names:
                    break
            self.satellites[satellite].is_producing = 0

    def get_system_state(self):
        total_queue_usage = {}
        total_computing_memory={}
        for node in self.satellite_names:
            average_usage = min(self.satellites[node].current_memory_occupy / self.memory,1)
            computing_memory = self.satellites[node].current_computing_queue_size / self.memory
            total_queue_usage[node] = average_usage
            total_computing_memory[node] = computing_memory
        return total_queue_usage

    def run(self, duration):
        if self.env.now==0:
            if self.use_global_task_generator:
                self.env.process(self.generate_global_traffic())
            else:
                for satellite in self.satellite_names:
                    self.env.process(self.generate_traffic(satellite))
            for satellite in self.satellites:
                self.satellites[satellite].all_start()
        self.env.run(until=self.env.now+duration)
        if 'Is_computing' in self.statics_datas:
            for satellite in self.satellites:
                self.statics_datas['Is_computing']+= self.satellites[satellite].is_computing

    def upgrade_all(self,graph,landmarks):

        self.landmarks = landmarks
        new_nodes = set(graph.nodes())
        old_nodes = set(self.graph.nodes())
        new_edges = set(graph.edges())
        old_edges = set(self.graph.edges())
        old_direct_satellites=self.direct_satellites

        self.satellite_names = [node for node in graph]
        self.graph = graph
        self.propagator.update(graph)
        self.propagator.reset_parameters()
        self.propagator.trans_parameters(self.max_hop,self.downstream_delays,self.reward_function)

        flattened_list = set(sum(self.landmarks.values(), []))
        new_direct_satellites = flattened_list
        self.direct_satellites = flattened_list
        for node in old_direct_satellites-new_direct_satellites:
            self.satellites[node].is_downlink=False
        for node in new_direct_satellites-old_direct_satellites:
            self.satellites[node].is_downlink = True

        for node in new_nodes - old_nodes:
            self.satellites[node] = Satellite_with_Computing(
                self.mode,
                self.select_mode,
                self.epsilon,
                self.max_hop,
                self.max_size,
                self.device,
                self.env,
                node,
                [neighbor for neighbor in self.graph.neighbors(node)],
                self.memory,
                self.computing_ability,
                self.transmission_rate,
                self.downlink_rate,
                self.state_update_period,
                True if node in self.direct_satellites else False,
                self.logger,
                self.statics_datas,
                heartbeat_timeout=self.heartbeat_timeout,
                task_profile=self.task_profile,
                scheme=self.scheme,
            )
            self.satellites[node].set_propagator(self.propagator)
            self.satellites[node].all_start()
            self.satellites[node].adjacency_table_exchanger()
            if not self.use_global_task_generator:
                self.env.process(self.generate_traffic(node))

        for node in old_nodes - new_nodes:
            self.env.process(self.satellites[node].self_missing())
            del self.satellites[node]
        for edge in new_edges - old_edges:
            node, neighbor = edge
            self.satellites[node].add_neighbor(neighbor)
            self.satellites[neighbor].add_neighbor(node)
        for satellite in self.satellites:
            self.satellites[satellite].trans_parameters(self.q_net,self.reward_function,self.direct_satellites)