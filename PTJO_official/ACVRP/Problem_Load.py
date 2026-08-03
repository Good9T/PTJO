import torch
import numpy as np
import os


def load_acvrp_file(file_path, device='cpu', normalize=True):

    with open(file_path, 'r') as f:
        lines = f.readlines()

    dimension = None
    capacity = None
    demand_section = False
    weight_section = False
    demand_values = []
    matrix_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            capacity = float(line.split(':')[1].strip())
        elif line == 'DEMAND_SECTION':
            demand_section = True
            weight_section = False
            continue
        elif line == 'EDGE_WEIGHT_SECTION':
            demand_section = False
            weight_section = True
            continue
        elif line == 'EOF':
            break
        elif demand_section and line:
            parts = line.split()
            if len(parts) == 2:
                demand_values.append(float(parts[1]))
        elif weight_section and line:
            matrix_lines.append(line)

    matrix_str = ' '.join(matrix_lines)
    matrix_flat = list(map(float, matrix_str.split()))
    matrix = np.array(matrix_flat).reshape(dimension, dimension)
    matrix = torch.tensor(matrix, dtype=torch.float, device=device)
    matrix[torch.arange(dimension), torch.arange(dimension)] = 0.0

    scaler = None
    if normalize:
        max_val = matrix.max()
        if max_val > 0:
            scaler = max_val.item()
            matrix = matrix / max_val

    demand = np.array(demand_values, dtype=np.float32)
    demand_norm = demand / capacity if capacity > 0 else demand

    demand_tensor = torch.tensor(demand_norm, dtype=torch.float, device=device)
    demand_expanded = demand_tensor.unsqueeze(0).unsqueeze(2).expand(1, dimension, dimension)

    problems = torch.stack([matrix.unsqueeze(0), demand_expanded], dim=3)
    trans_problems = torch.transpose(problems, 1, 2)
    node_demand = demand_tensor.unsqueeze(0)
    node_size = dimension

    return problems, node_demand, trans_problems, node_size, scaler
