import os
import sys
import threading
import pty
import re
from contextlib import contextmanager
from functools import wraps
import torch


class TrainingState:
    """A simple class to encapsulate all scalar training states that need to be saved."""
    def __init__(self, epoch: int = 0, update_step: int = 0, global_step: int = 0):
        self.epoch = epoch
        self.update_step = update_step
        self.global_step = global_step

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "update_step": self.update_step,
            "global_step": self.global_step,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.update_step = state_dict["update_step"]
        self.global_step = state_dict["global_step"]


@contextmanager
def tee_output_to_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log_file = open(path, "ab", buffering=0)

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    read_fd = None
    try:
        master_fd, slave_fd = pty.openpty()
        os.dup2(slave_fd, stdout_fd)
        os.dup2(slave_fd, stderr_fd)
        os.close(slave_fd)
        read_fd = master_fd
    except Exception:
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, stdout_fd)
        os.dup2(write_fd, stderr_fd)
        os.close(write_fd)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass

    ansi_escape = re.compile(rb"\x1B\[[0-?]*[ -/]*[@-~]")

    def _reader():
        while True:
            try:
                data = os.read(read_fd, 4096)
            except OSError:
                break
            if not data:
                break
            os.write(saved_stdout_fd, data)
            log_file.write(ansi_escape.sub(b"", data))
            log_file.flush()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        try:
            os.close(read_fd)
        except Exception:
            pass
        reader_thread.join(timeout=1)
        log_file.flush()
        log_file.close()


def capture_output_to_training_log(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        log_path = os.path.join(self.output_dir, "training.log")
        with tee_output_to_file(log_path):
            return func(self, *args, **kwargs)
    return wrapper


def params_l2_norm(params):
    params = [p for p in params if p is not None]
    if not params:
        return 0.0
    device = params[0].device
    total = torch.zeros((), device=device, dtype=torch.float32)
    for param in params:
        total += param.detach().float().pow(2).sum()
    return torch.sqrt(total).item()


class FullMemoryTracker:
    def __init__(self, model):
        self.model = model
        self.stats = {}
        self.hooks = []

    def _get_tensor_mem(self, tensor):
        if torch.is_tensor(tensor):
            return tensor.element_size() * tensor.nelement() / (1024**2)
        return 0

    def hook_before(self, module, input, name):
        # 记录进入该层前的显存状态
        torch.cuda.synchronize()  # 强制对齐，确保测得准，但会变慢
        module._mem_before = torch.cuda.memory_allocated()

    def hook_after(self, module, input, output, name):
        # 记录离开该层后的显存状态
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        # 计算该层运行期间增加的显存（这包含了激活值和中间变量）
        diff = (mem_after - module._mem_before) / (1024**2)

        # 计算参数和梯度的大小
        param_mem = sum(p.element_size() * p.nelement() for p in module.parameters(recurse=False)) / (1024**2)
        grad_mem = sum(p.grad.element_size() * p.grad.nelement() if p.grad is not None else 0
                       for p in module.parameters(recurse=False)) / (1024**2)

        if name not in self.stats:
            self.stats[name] = {'param': param_mem, 'peak_delta': 0, 'grad': 0, 'output': 0}

        self.stats[name]['peak_delta'] = max(self.stats[name]['peak_delta'], diff)
        self.stats[name]['grad'] = max(self.stats[name]['grad'], grad_mem)
        self.stats[name]['output'] = max(self.stats[name]['output'], self._get_tensor_mem(output))

    def track(self):
        for name, module in self.model.named_modules():
            # 过滤掉层级太深的，看主要的 Block 即可
            if len(list(module.children())) <= 3:
                h_pre = module.register_forward_pre_hook(lambda m, i, n=name: self.hook_before(m, i, n))
                h_post = module.register_forward_hook(lambda m, i, o, n=name: self.hook_after(m, i, o, n))
                self.hooks.extend([h_pre, h_post])

    def report(self):
        print(f"\n{'Module Name':<50} | {'Param(MB)':<10} | {'Grad(MB)':<10} | {'Output(MB)':<12} | {'Net-Delta(MB)':<12}")
        print("-" * 105)
        # 按增量排序，找出真正的显存大户
        sorted_items = sorted(self.stats.items(), key=lambda x: x[1]['peak_delta'], reverse=True)
        for name, s in sorted_items[:200]:
            print(f"{name[:50]:<50} | {s['param']:>10.1f} | {s['grad']:>10.1f} | {s['output']:>12.1f} | {s['peak_delta']:>12.1f}")
        print(f"Total memory usage: {sum(s['peak_delta'] for s in self.stats.values()):.1f} MB")

    def stop(self):
        for h in self.hooks:
            h.remove()

