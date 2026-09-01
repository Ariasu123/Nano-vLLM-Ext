# 纯逻辑 CPU 单测的导入垫片。
#
# 背景：`import nanovllm` 会执行 nanovllm/__init__.py → LLM → llm_engine → model_runner
# → attention，attention.py 在模块级 `import triton` / `@triton.jit` / `from flash_attn import ...`，
# llm_engine.py 还 `from transformers import AutoTokenizer`。这些都是 GPU 专用依赖，
# 在无 CUDA 的本地（如 macOS）装不上。
#
# scheduler / block_manager / metrics / rejection_sampler 等纯逻辑测试并不真正调用这些内核，
# 只是被 import 链带进来。这里在导入 nanovllm 之前，往 sys.modules 注入极简 stub，
# 让包能在 CPU 上 import；stub 里的函数永远不会被这些测试调用（真正跑模型在 GPU 服务器上）。
import sys
import types

# `@torch.compile`（sampler.py）在被调用时会导入 torch._dynamo/_inductor，进而深入访问 triton
# 内部子模块（triton.backends.compiler 等），stub 无法满足。CPU 单测不跑编译路径，这里把
# torch.compile 换成透传装饰器，彻底避开 dynamo/inductor→triton 的导入链。必须在 import nanovllm 之前执行。
try:
    import torch
    torch.compile = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
except ImportError:
    pass


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# triton：模块级用到 @triton.jit 装饰器与 triton.language.constexpr 注解。
# jit 直接返回原函数即可（kernel 体不会在 CPU 测试中被调用）。
if "triton" not in sys.modules:
    triton = _stub("triton", jit=lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f)))
    triton.cdiv = lambda a, b: (a + b - 1) // b
    triton.__version__ = "0.0.0-stub"
    # torch._dynamo 在 import 时若发现 triton 可导入，会访问 triton.language.dtype 等属性；
    # 这里补上占位属性，避免 AttributeError（真实内核永不在 CPU 测试中执行）。
    tl = _stub("triton.language", constexpr=object(), dtype=type("dtype", (), {}))
    triton.language = tl

# flash_attn：只需存在同名可调用符号，供 attention.py 的 from-import 成功。
if "flash_attn" not in sys.modules:
    def _flash_unavailable(*a, **k):  # 若被调用说明测试误入 GPU 路径
        raise RuntimeError("flash_attn stub called in CPU test")
    _stub("flash_attn",
          flash_attn_varlen_func=_flash_unavailable,
          flash_attn_with_kvcache=_flash_unavailable)

# transformers：llm_engine / config 只在 import 时引用 AutoTokenizer / AutoConfig，
# 纯逻辑测试用 SimpleNamespace mock config，不会真正构造它们。
if "transformers" not in sys.modules:
    class _Auto:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise RuntimeError("transformers stub called in CPU test")
    class _Qwen3Config:  # 仅供 qwen3.py 的 from-import 成功；CPU 测试不构造模型
        pass
    _stub("transformers", AutoTokenizer=_Auto, AutoConfig=_Auto, Qwen3Config=_Qwen3Config)
