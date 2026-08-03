import torch
import numpy as np
import os


def load_atsp_file(file_path, device='cpu', normalize=True):

    with open(file_path, 'r') as f:
        lines = f.readlines()

    dimension = None
    edge_weight_section = False
    matrix_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        if line.startswith('EDGE_WEIGHT_SECTION'):
            edge_weight_section = True
            continue
        if edge_weight_section and line != 'EOF':
            matrix_lines.append(line)
        if line == 'EOF':
            break

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

    problems = matrix.unsqueeze(0)  # (1, node_size, node_size)
    trans_problems = torch.transpose(problems, 1, 2)
    node_size = dimension

    return problems, trans_problems, node_size, scaler


