import subprocess

import torch
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


def validar_cuda():
    """Valida si CUDA está disponible para PyTorch y si Ollama usa la GPU."""
    print("=" * 60)
    print("VALIDACIÓN CUDA / GPU")
    print("=" * 60)

    # 1. PyTorch + CUDA
    print(f"\n[PyTorch]  versión: {torch.__version__}")
    print(f"[PyTorch]  CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[PyTorch]  CUDA versión: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            print(f"[PyTorch]  GPU {i}: {torch.cuda.get_device_name(i)}")
        # prueba rápida en GPU
        x = torch.rand(3, 3, device="cuda")
        y = torch.rand(3, 3, device="cuda")
        _ = x @ y
        print("[PyTorch]  Operación en GPU: OK")
    else:
        print("[PyTorch]  ⚠ CUDA NO disponible — PyTorch usará CPU")

    # 2. nvidia-smi (procesos GPU)
    print()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_gpu_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("[nvidia-smi] Procesos usando GPU:")
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")
            if "ollama" in result.stdout.lower():
                print("[nvidia-smi] ✓ Ollama está usando la GPU")
            else:
                print("[nvidia-smi] ⚠ Ollama no aparece en los procesos GPU")
        else:
            print("[nvidia-smi] No se detectaron procesos usando la GPU")
    except FileNotFoundError:
        print("[nvidia-smi] No disponible en el sistema")

    # 3. Verificar Ollama GPU via modelo cargado
    print()
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("[Ollama ps] Modelos cargados:")
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")
            stdout_lower = result.stdout.lower()
            if "gpu" in stdout_lower or "cuda" in stdout_lower:
                print("[Ollama]  ✓ Modelo ejecutándose en GPU")
            elif "cpu" in stdout_lower:
                print("[Ollama]  ⚠ Modelo ejecutándose en CPU, no en GPU")
        else:
            print("[Ollama ps] No hay modelos cargados actualmente")
    except FileNotFoundError:
        print("[Ollama]  CLI de ollama no encontrada")

    print("\n" + "=" * 60 + "\n")


# Herramienta 1: búsqueda simulada en catálogo
@tool
def buscar_producto(nombre: str) -> str:
    """Busca un producto por nombre y devuelve información resumida."""
    catalogo = {
        "teclado": {"precio": 49.99, "stock": 12},
        "raton": {"precio": 19.99, "stock": 0},
        "monitor": {"precio": 199.99, "stock": 4},
    }

    key = nombre.strip().lower()
    if key not in catalogo:
        return f"No encontré el producto '{nombre}'."

    p = catalogo[key]
    return f"Producto: {nombre}. Precio: {p['precio']} USD. Stock: {p['stock']}."


# Herramienta 2: cálculo simple de descuento
@tool
def calcular_total(precio: float, cantidad: int, descuento_pct: float = 0) -> str:
    """Calcula el total aplicando descuento porcentual."""
    subtotal = precio * cantidad
    total = subtotal * (1 - descuento_pct / 100)
    return f"Subtotal: {subtotal:.2f} USD. " f"Descuento: {descuento_pct:.1f}%. " f"Total: {total:.2f} USD."


def main():
    validar_cuda()

    print("Inicializando...")
    # Modelo local vía Ollama
    model = ChatOllama(
        model="gemma4:26b",
        temperature=0,
    )

    print("Creando agentes...")
    # Agente con herramientas
    agent = create_agent(
        model=model,
        tools=[buscar_producto, calcular_total],
        system_prompt=(
            "Eres un asistente de ventas. "
            "Usa herramientas cuando necesites datos exactos o cálculos. "
            "Si no sabes algo, dilo claramente."
        ),
    )

    print("Ejecutando ejemplos...")
    # Ejemplo 1
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": ("Busca el producto teclado y dime cuánto costarían " "3 unidades con un descuento del 10%."),
                }
            ]
        }
    )

    print("=== RESPUESTA 1 ===")
    print(response)

    # Ejemplo 2
    response2 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": ("¿Hay stock de raton? " "Si no hay, dímelo claramente."),
                }
            ]
        }
    )

    print("\n=== RESPUESTA 2 ===")
    print(response2)


if __name__ == "__main__":
    main()
