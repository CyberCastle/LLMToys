#!/usr/bin/env python3
import shutil
import subprocess
import sys
import ctypes
from pathlib import Path


def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_nvidia_smi():
    print_section("1. Verificando driver NVIDIA con nvidia-smi")
    if not shutil.which("nvidia-smi"):
        print("FALLO: 'nvidia-smi' no está en el PATH.")
        return False

    code, out, err = run_cmd(["nvidia-smi"])
    if code != 0:
        print("FALLO: 'nvidia-smi' devolvió error.")
        print("stderr:", err)
        return False

    print("OK: 'nvidia-smi' funciona.")
    print(out[:800])
    return True


def check_nvcc():
    print_section("2. Verificando toolkit CUDA con nvcc")
    if not shutil.which("nvcc"):
        print("ADVERTENCIA: 'nvcc' no está en el PATH.")
        print("Esto puede significar que el toolkit CUDA no está instalado o no está exportado.")
        return False

    code, out, err = run_cmd(["nvcc", "--version"])
    if code != 0:
        print("FALLO: 'nvcc --version' devolvió error.")
        print("stderr:", err)
        return False

    print("OK: 'nvcc' está disponible.")
    print(out)
    return True


def check_cuda_runtime_lib():
    print_section("3. Verificando librerías CUDA")
    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so.12",
    ]

    for lib in candidates:
        try:
            ctypes.CDLL(lib)
            print(f"OK: se pudo cargar {lib}")
            return True
        except OSError:
            continue

    print("ADVERTENCIA: no se pudo cargar libcudart.")
    print("Puede faltar CUDA runtime o LD_LIBRARY_PATH.")
    return False


def check_torch_cuda():
    print_section("4. Verificando CUDA desde PyTorch (opcional)")
    try:
        import torch
    except ImportError:
        print("PyTorch no está instalado. Se omite esta prueba.")
        return None

    print(f"PyTorch versión: {torch.__version__}")
    print(f"CUDA compilada en PyTorch: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("FALLO: torch.cuda.is_available() = False")
        return False

    count = torch.cuda.device_count()
    print(f"OK: CUDA disponible en PyTorch. GPUs detectadas: {count}")

    for i in range(count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    try:
        x = torch.rand(3, 3).cuda()
        y = torch.rand(3, 3).cuda()
        z = x @ y
        print("OK: operación en GPU ejecutada correctamente.")
        print(z)
        return True
    except Exception as e:
        print("FALLO: PyTorch detecta CUDA pero la operación en GPU falló.")
        print(str(e))
        return False


def main():
    print("Validación de instalación CUDA\n")

    results = {
        "nvidia_smi": check_nvidia_smi(),
        "nvcc": check_nvcc(),
        "cudart": check_cuda_runtime_lib(),
        "torch": check_torch_cuda(),
    }

    print_section("Resumen")
    for key, value in results.items():
        print(f"{key}: {value}")

    if results["nvidia_smi"] and (results["nvcc"] or results["cudart"]):
        print("\nRESULTADO GENERAL: CUDA/driver parece estar instalado correctamente.")
        sys.exit(0)
    else:
        print("\nRESULTADO GENERAL: hay indicios de instalación incompleta o PATH/librerías mal configuradas.")
        sys.exit(1)


if __name__ == "__main__":
    main()
