"""V21 шаг 0: проверка GPU/окружения для обучения cross-encoder.

Фиксирует фактами (а не предположениями): есть ли GPU-устройство, драйвер, CUDA в torch,
память, CPU/RAM/диск и версии библиотек. Если GPU нет — это блокер обучения (шаг 3), но
подготовка цепочки (данные, модель, экспорт, скорость) продолжается.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(command, timeout=20):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, shell=True)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip()[:2000], "stderr": result.stderr.strip()[:500]}
    except Exception as error:  # noqa: BLE001 - диагностика
        return {"command": command, "returncode": None, "error": f"{type(error).__name__}: {error}"}


def meminfo():
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
        values = {}
        for line in lines:
            key, _, rest = line.partition(":")
            values[key] = rest.strip()
        total_kb = int(values.get("MemTotal", "0 kB").split()[0])
        available_kb = int(values.get("MemAvailable", "0 kB").split()[0])
        return {"mem_total_gb": round(total_kb / 1024 / 1024, 2),
                "mem_available_gb": round(available_kb / 1024 / 1024, 2)}
    except Exception:  # noqa: BLE001
        return {}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/gpu_check_v21.json")
    parser.add_argument("--min_gpu_memory_gb", type=float, default=8.0)
    args = parser.parse_args(argv)

    checks = {
        "nvidia_smi": run("nvidia-smi"),
        "nvidia_smi_query": run("nvidia-smi --query-gpu=name,memory.total,driver_version "
                                "--format=csv,noheader"),
        "device_nodes": sorted(str(path) for path in Path("/dev").glob("nvidia*")),
        "lspci_gpu": run("lspci | grep -iE 'vga|3d|nvidia'"),
    }

    torch_info = {"installed": False}
    try:
        import torch
        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())],
            "gpu_memory_gb": [round(torch.cuda.get_device_properties(index).total_memory / 1024 ** 3, 2)
                              for index in range(torch.cuda.device_count())],
            "threads": torch.get_num_threads(),
        }
    except Exception as error:  # noqa: BLE001
        torch_info = {"installed": False, "error": f"{type(error).__name__}: {error}"}

    libraries = {}
    for name in ("numpy", "transformers", "onnx", "onnxruntime", "sklearn"):
        try:
            module = __import__(name if name != "sklearn" else "sklearn")
            libraries[name] = getattr(module, "__version__", "?")
        except Exception as error:  # noqa: BLE001
            libraries[name] = f"нет: {type(error).__name__}"

    cpu = {"cores": os.cpu_count(),
           "model": next((line.split(":", 1)[1].strip() for line in
                          Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), "?")}
    disk = shutil.disk_usage(ROOT)
    gpu_present = bool(checks["device_nodes"]) or checks["nvidia_smi"].get("returncode") == 0
    enough_memory = bool(torch_info.get("gpu_memory_gb")) and \
        min(torch_info["gpu_memory_gb"]) >= args.min_gpu_memory_gb

    report = {
        "rule": "шаг 0 v21: проверка GPU перед обучением cross-encoder",
        "platform": {"python": platform.python_version(), "machine": platform.machine(),
                     "system": platform.system()},
        "hardware": {"cpu": cpu, "memory": meminfo(),
                     "disk_free_gb": round(disk.free / 1024 ** 3, 2)},
        "checks": checks,
        "torch": torch_info,
        "libraries": libraries,
        "requirements": {"min_gpu_memory_gb": args.min_gpu_memory_gb,
                         "cuda_needed": True, "torch_cuda_needed": True},
        "verdict": {
            "gpu_present": gpu_present,
            "torch_cuda_available": bool(torch_info.get("cuda_available")),
            "gpu_memory_ok": enough_memory,
            "can_train_cross_encoder": bool(gpu_present and torch_info.get("cuda_available")
                                            and enough_memory),
            "blocker": None if (gpu_present and torch_info.get("cuda_available") and enough_memory)
            else "в песочнице нет GPU: nvidia-smi отсутствует, /dev/nvidia* нет, "
                 "torch.cuda.is_available() = False → обучение cross-encoder на GPU невозможно",
        },
        "what_is_possible_on_cpu": [
            "подготовка данных и статистики (ce_data_v21.json)",
            "сборка архитектуры cross-encoder на реальных весах rubert-tiny2",
            "smoke-обучение на CPU для замера секунд/шаг и проверки корректности пайплайна",
            "экспорт в ONNX + int8-квантование + замер CPU-скорости (шаг 4)",
        ],
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(report["verdict"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
