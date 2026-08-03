import torch
import numpy as np
import os


def load_atsp_file(file_path, device='cpu', normalize=True):
    """
    从 .atsp 文件读取距离矩阵

    Args:
        file_path: .atsp 文件路径
        device: 'cpu' 或 'cuda'
        normalize: 是否归一化到 [0, 1]

    Returns:
        problems: (1, node_size, node_size) 距离矩阵
        trans_problems: (1, node_size, node_size) 转置矩阵
        node_size: 节点数
        scaler: 归一化时的最大值（用于还原）
    """
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

    # 解析矩阵
    matrix_str = ' '.join(matrix_lines)
    matrix_flat = list(map(float, matrix_str.split()))
    matrix = np.array(matrix_flat).reshape(dimension, dimension)
    matrix = torch.tensor(matrix, dtype=torch.float, device=device)

    # ✅ 第一步：对角线置0（消除9999影响）
    matrix[torch.arange(dimension), torch.arange(dimension)] = 0.0

    # 归一化
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


if __name__ == "__main__":
    dataset_dir = r"E:\PycharmProjcets\PTJO\ATSP_HP_dataset"
    if not os.path.exists(dataset_dir):
        print(f"目录 {dataset_dir} 不存在")
    else:
        atsp_files = [f for f in os.listdir(dataset_dir) if f.endswith('.atsp')]

        if not atsp_files:
            print("未找到 .atsp 文件")
        else:
            file_path = os.path.join(dataset_dir, atsp_files[0])
            print(f"读取文件: {file_path}")

            problems, trans_problems, node_size, scaler = load_atsp_file(file_path, normalize=True)

            print(f"节点数量: {node_size}")
            print(f"problems shape: {problems.shape}")
            print(f"trans_problems shape: {trans_problems.shape}")
            print(f"scaler (归一化分母): {scaler}")
            print(f"归一化后最大边: {problems.max().item():.4f}")
            print(f"归一化后最小边: {problems.min().item():.4f}")
            print(
                f"对角线是否为0: {torch.all(problems[0, torch.arange(node_size), torch.arange(node_size)] == 0).item()}")