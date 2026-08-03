from dataclasses import dataclass
import torch
from Problem_Load import load_acvrp_file

@dataclass
class Reset_State:
    problems: torch.Tensor
    start_node_idx: torch.Tensor

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    selected_count: int = None
    current_node: torch.Tensor = None
    state_mask: torch.Tensor = None
    finished: torch.Tensor = None
    cur_dist: torch.Tensor = None

class Env:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.pomo_size = None
        self.node_size = None
        self.scaler = None

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.problems = None
        self.node_demand = None
        self.trans_problems = None
        self.node_coords = None

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.load = None
        self.at_depot = None
        self.step_state = None
        self.mask = None
        self.finished = None
        self.node_demand = None
        self.visited_flag = None

    def load_problems(self, file_path, device='cpu', aug_enable=False, aug_factor=1):
        self.batch_size = 1
        (
            self.problems,
            self.node_demand,
            self.trans_problems,
            self.node_size,
            self.scaler
        ) = load_acvrp_file(
            file_path,
            device
        )
        if aug_enable:
            self.problems = self.problems.repeat(aug_factor, 1, 1, 1)
            self.node_demand = self.node_demand.repeat(aug_factor, 1)
            self.trans_problems = self.trans_problems.repeat(aug_factor, 1, 1, 1)
            self.batch_size = self.batch_size * aug_factor
        self.pomo_size = self.node_size - 1
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def load_trans_problems(self):
        self.problems = self.trans_problems

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        self.load = torch.zeros(size=(self.batch_size, self.pomo_size))
        self.at_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        self.visited_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_size))
        self.mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_size))
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        start_node_idx = torch.arange(start=1, end=self.pomo_size+1)[None, :].expand(self.batch_size, self.pomo_size)
        return Reset_State(problems=self.problems, start_node_idx=start_node_idx), None, False

    def pre_step(self):
        reward = None
        done = False
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.state_mask = torch.zeros(self.batch_size, self.pomo_size, self.node_size)
        self.step_state.finished = self.finished
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        self.at_depot = (selected == 0)
        demand_list = self.node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand

        self.load[self.at_depot] = 1

        self.visited_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, node)
        self.visited_flag[:, :, 0][~self.at_depot] = 0
        self.mask = self.visited_flag.clone()
        round_error_epsilon = 0.00001
        demand_unfeasible = self.load[:, :, None] + round_error_epsilon < demand_list
        self.mask[demand_unfeasible] = float('-inf')
        new_finished = (self.visited_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + new_finished

        self.mask[:, :, 0][self.finished] = 0
        self._update_step_state()

        done = self.finished.all()
        if done:
            reward = -self._get_total_distance()
        else:
            reward = None
        return self.step_state, reward, done



    def _update_step_state(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.state_mask = self.mask
        self.step_state.finished = self.finished
        self.step_state.cur_dist = self._get_cur_dist()


    def _get_total_distance(self):
        node_from = self.selected_node_list
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.selected_count)
        selected_cost = self.problems[batch_index, node_from, node_to, 0].squeeze(dim=-1)
        total_distance = selected_cost.sum(2) * self.scaler

        return total_distance

    def _get_cur_dist(self):
        if self.current_node is None:
            return None
        dist_matrix = self.problems[..., 0]
        cur_dist = dist_matrix[self.BATCH_IDX, self.current_node, :]
        return cur_dist