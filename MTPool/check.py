# 创建 check_pyg_api.py
import torch_geometric

print(f"PyTorch Geometric version: {torch_geometric.__version__}")

# 检查 topk_pool 模块
try:
    import torch_geometric.nn.pool.topk_pool as topk_pool
    print("topk_pool functions:", [x for x in dir(topk_pool) if not x.startswith('_')])
except Exception as e:
    print("topk_pool error:", e)

# 检查 utils 模块
try:
    import torch_geometric.utils as utils
    available_funcs = [x for x in dir(utils) if not x.startswith('_')]
    filter_funcs = [x for x in available_funcs if 'filter' in x.lower()]
    print("utils filter functions:", filter_funcs)
    print("All utils functions:", len(available_funcs), "functions available")
except Exception as e:
    print("utils error:", e)

# 检查 nn.pool 模块
try:
    import torch_geometric.nn.pool as pool
    print("pool functions:", [x for x in dir(pool) if not x.startswith('_')])
except Exception as e:
    print("pool error:", e)