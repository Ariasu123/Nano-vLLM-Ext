# 模型类只创建了形状正确的空 Parameter，本文件负责从 .safetensors 文件填入真实权重。

# 【为什么不能全部直接 copy】
# 本实现使用张量并行和融合层：
# - 完整权重要按 rank 切片；
# - checkpoint 中分开的 q_proj/k_proj/v_proj 要写入一个 qkv_proj；
# - gate_proj/up_proj 要写入一个 gate_up_proj。
# 因此每种并行层会把自己的 weight_loader 方法挂到 Parameter 上。
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


# 普通参数的默认加载方法：整块原地复制。
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# model 是已经创建好的空模型；path 是包含 safetensors 的模型目录。
def load_model(model: nn.Module, path: str):
    # getattr(obj,name,default) 安全读取属性；没有映射时使用空字典。
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # glob 找到目录下所有 .safetensors；大模型通常被拆成多个 shard（分片）文件。
    for file in glob(os.path.join(path, "*.safetensors")):

        # with 是上下文管理器语法：退出代码块时自动关闭文件。
        # "pt" 表示以 PyTorch Tensor 读取，"cpu" 表示先从磁盘加载到 CPU。
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 先检查这个名称是否属于融合层。
                # 例：model.layers.0.self_attn.q_proj.weight 包含 "q_proj"。
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # v 是替换后的融合层名，shard_id 表示写入融合参数的哪一部分。
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)

                        # 根据字符串形式的完整名称找到模型 Parameter。
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")

                        # 融合层加载器会同时做“选择当前 rank 分片”和“选择 Q/K/V 子区域”。
                        weight_loader(param, f.get_tensor(weight_name), shard_id)

                        # 去处理下一个权重
                        break
                else:
                    # 这是 Python for...else：只有循环没有 break 时才进入 else，
                    # 即当前权重不是需要特殊处理的融合权重。
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
