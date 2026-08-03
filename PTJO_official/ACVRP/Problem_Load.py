import torch
import numpy as np
import os


def load_acvrp_file(file_path, device='cpu', normalize=True):
    """
    从 .dat 文件读取 ACVRP 实例（Fischetti格式，含需求和容量）

    Args:
        file_path: .dat 文件路径
        device: 'cpu' 或 'cuda'
        normalize: 是否归一化距离到 [0, 1]

    Returns:
        problems: (1, node_size, node_size, 2) 距离+需求矩阵
        node_demand: (1, node_size) 归一化需求
        trans_problems: (1, node_size, node_size, 2) 转置矩阵
        scaler: 距离最大值（用于还原）
        capacity: 车辆容量
        node_size: 节点数
    """
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

    # 距离矩阵
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

    # 需求
    demand = np.array(demand_values, dtype=np.float32)
    demand_norm = demand / capacity if capacity > 0 else demand

    demand_tensor = torch.tensor(demand_norm, dtype=torch.float, device=device)
    demand_expanded = demand_tensor.unsqueeze(0).unsqueeze(2).expand(1, dimension, dimension)

    problems = torch.stack([matrix.unsqueeze(0), demand_expanded], dim=3)
    trans_problems = torch.transpose(problems, 1, 2)
    node_demand = demand_tensor.unsqueeze(0)
    node_size = dimension

    return problems, node_demand, trans_problems, node_size, scaler


if __name__ == "__main__":
    import os

    dataset_dir = r"E:\PycharmProjcets\PTJO\ACVRP_dataset_from_ATSP"

    if not os.path.exists(dataset_dir):
        print(f"目录 {dataset_dir} 不存在")
    else:
        files = [f for f in os.listdir(dataset_dir) if f.endswith('.dat')]
        if not files:
            print("未找到 .dat 文件")
        else:
            file_path = os.path.join(dataset_dir, files[0])
            print(f"读取文件: {file_path}")

            problems, node_demand, trans_problems, scaler, capacity, node_size = load_acvrp_file(file_path)

            print(f"节点数量: {node_size}")
            print(f"车辆容量: {capacity}")
            print(f"problems shape: {problems.shape}")
            print(f"trans_problems shape: {trans_problems.shape}")
            print(f"node_demand shape: {node_demand.shape}")
            print(f"scaler (归一化分母): {scaler}")
            print(f"归一化后距离最大值: {problems[0, :, :, 0].max().item():.4f}")
            print(f"归一化后距离最小值: {problems[0, :, :, 0].min().item():.4f}")
            print(f"需求最大值: {node_demand.max().item():.4f}")
            print(f"需求最小值: {node_demand.min().item():.4f}")
            diag = problems[0, torch.arange(node_size), torch.arange(node_size), 0]
            print(f"距离对角线是否为0: {torch.all(diag == 0).item()}")

            total_demand = node_demand[0, 1:].sum().item() * capacity
            print(f"总需求 (不含仓库): {total_demand:.2f}")
            print(f"需求/容量: {total_demand / capacity:.2f} 倍")