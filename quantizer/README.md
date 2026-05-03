## Quantizador Genérico

Este módulo implementa un cuantizador reutilizable para modelos Causal LM de Hugging Face usando AWQ W4A16 y GPTQ W4A16. El objetivo es producir checkpoints en formato `compressed-tensors` que luego puedan validarse o desplegarse con `vllm` cuando exista un entorno compatible.

## Requisitos

- Python `>=3.12,<3.14.1`.
- GPU NVIDIA con CUDA y suficiente VRAM para cargar el modelo base con offload a CPU.
- `uv` para instalar el subproyecto aislado.
- Autenticación de Hugging Face si el modelo o el dataset son gated.

## Variables de Entorno

### Compartidas

| Variable                            | Defecto         | Descripcion                                                                                                                       |
| ----------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `QUANTIZER_MODEL_ID`                | requerido       | Identificador del modelo Causal LM en Hugging Face.                                                                               |
| `QUANTIZER_CALIBRATION_DATASET`     | requerido       | Dataset de calibracion en Hugging Face.                                                                                           |
| `QUANTIZER_CALIBRATION_SPLIT`       | `train`         | Split del dataset.                                                                                                                |
| `QUANTIZER_DATASET_CONFIG_NAME`     | vacio           | Configuracion o subset opcional del dataset.                                                                                      |
| `QUANTIZER_MAX_SEQUENCE_LENGTH`     | `2048`          | Longitud maxima tras tokenizacion.                                                                                                |
| `QUANTIZE_SCHEME`                   | `both`          | `awq`, `gptq` o `both`.                                                                                                           |
| `QUANTIZER_OUTPUT_DIR`              | `out/quantized` | Directorio base donde se guardan los artefactos.                                                                                  |
| `QUANTIZER_TRUST_REMOTE_CODE_MODEL` | `false`         | Permite `trust_remote_code` al cargar modelo y tokenizer.                                                                         |
| `QUANTIZER_DATASET_TEMPLATES_PATH`  | bundled         | Ruta alternativa al YAML de plantillas de dataset.                                                                                |
| `QUANTIZER_RUN_VLLM_SMOKE_TEST`     | `false`         | Ejecuta la prueba de humo de `vllm` si existe un entorno compatible.                                                              |
| `QUANTIZER_MEMORY_PREFLIGHT_MODE`   | `guard`         | Controla el preflight de memoria antes de cuantizar: `off`, `report`, `guard` o `fail-fast`.                                      |
| `QUANTIZER_SKIP_MEMORY_PREFLIGHT`   | interno         | Omite el preflight en subprocesos ya validados por el proceso padre. El runner lo usa internamente cuando `QUANTIZE_SCHEME=both`. |

### AWQ

| Variable                                        | Defecto     | Descripcion                                                                                     |
| ----------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------- |
| `QUANTIZER_AWQ_NUM_CALIBRATION_SAMPLES`         | `128`       | Numero de muestras de calibracion para AWQ.                                                     |
| `QUANTIZER_AWQ_MAX_GPU_MEMORY_GIB`              | `13.0`      | Presupuesto de VRAM para cargar el modelo antes del smoothing AWQ.                              |
| `QUANTIZER_AWQ_SEQUENTIAL_ONLOADING`            | `true`      | Activa el pipeline secuencial con offload conservador para AWQ.                                 |
| `QUANTIZER_AWQ_SEQUENTIAL_TARGETS`              | `safe-auto` | Targets del pipeline secuencial AWQ. Acepta `safe-auto`, `auto` o una lista separada por comas. |
| `QUANTIZER_AWQ_SEQUENTIAL_TARGETS_PER_SUBGRAPH` | `1`         | Cantidad de targets por subgrafo secuencial en AWQ. Valores mayores usan mas VRAM.              |
| `QUANTIZER_AWQ_MAPPINGS_PATH`                   | bundled     | Ruta alternativa al YAML de mappings AWQ.                                                       |

### GPTQ

| Variable                                         | Defecto     | Descripcion                                                                                       |
| ------------------------------------------------ | ----------- | ------------------------------------------------------------------------------------------------- |
| `QUANTIZER_GPTQ_NUM_CALIBRATION_SAMPLES`         | `512`       | Numero de muestras de calibracion para GPTQ.                                                      |
| `QUANTIZER_GPTQ_MAX_GPU_MEMORY_GIB`              | `10.0`      | Presupuesto de VRAM para cargar el modelo antes de acumular Hessians e invertir Cholesky en GPTQ. |
| `QUANTIZER_GPTQ_SEQUENTIAL_ONLOADING`            | `true`      | Activa el pipeline secuencial con offload conservador para GPTQ.                                  |
| `QUANTIZER_GPTQ_SEQUENTIAL_TARGETS`              | `safe-auto` | Targets del pipeline secuencial GPTQ. Acepta `safe-auto`, `auto` o una lista separada por comas.  |
| `QUANTIZER_GPTQ_SEQUENTIAL_TARGETS_PER_SUBGRAPH` | `1`         | Cantidad de targets por subgrafo secuencial en GPTQ. Valores mayores usan mas VRAM.               |

## Uso Rápido

AWQ:

```bash
QUANTIZER_MODEL_ID=org/modelo \
QUANTIZER_CALIBRATION_DATASET=org/dataset \
QUANTIZE_SCHEME=awq \
uv run quantizer/run.py
```

GPTQ:

```bash
QUANTIZER_MODEL_ID=org/modelo \
QUANTIZER_CALIBRATION_DATASET=org/dataset \
QUANTIZE_SCHEME=gptq \
uv run quantizer/run.py
```

Ambos esquemas:

```bash
QUANTIZER_MODEL_ID=org/modelo \
QUANTIZER_CALIBRATION_DATASET=org/dataset \
QUANTIZE_SCHEME=both \
uv run quantizer/run.py
```

Cuando `QUANTIZE_SCHEME=both`, el runner ejecuta AWQ y GPTQ en procesos separados para liberar la VRAM entre un esquema y el siguiente.

## Preflight de Memoria

El runner puede estimar el riesgo de OOM antes de cargar el modelo y empezar la calibracion. Esto se controla con `QUANTIZER_MEMORY_PREFLIGHT_MODE`:

- `off`: no ejecuta el preflight.
- `report`: imprime el diagnostico pero no bloquea la corrida.
- `guard`: bloquea solo si la corrida se estima inviable.
- `fail-fast`: bloquea tanto corridas `risky` como `inviable`.

Cuando `QUANTIZE_SCHEME=both`, el preflight corre en el proceso padre y los hijos reciben `QUANTIZER_SKIP_MEMORY_PREFLIGHT=true` para no duplicar el chequeo.

Si quieres intentar la validacion con `vllm`, actívala explícitamente:

```bash
QUANTIZER_RUN_VLLM_SMOKE_TEST=true \
uv run quantizer/run.py
```

## Agregar Soporte para una Nueva Arquitectura

1. Inspecciona `AutoConfig.from_pretrained(model_id).model_type`.
2. Agrega una nueva clave al archivo `assets/awq_mappings.yaml`.
3. Declara la lista de `smooth_layer` y `target_layers` usando regex sobre los nombres de módulo.
4. Vuelve a ejecutar el cuantizador. No hace falta modificar Python mientras el `model_type` exista en el YAML.

## Agregar Soporte para un Nuevo Dataset

1. Si el dataset ya expone una columna `text`, no hace falta registrar nada.
2. Si el dataset usa columnas estructuradas, agrega una entrada en `assets/dataset_templates.yaml`.
3. Usa `messages_template` para datasets tipo chat o `text_field` para datasets con una sola columna textual.
4. Los placeholders opcionales se escriben como `{campo?}` y se omiten línea completa si el valor viene vacío.

## Artefactos Generados

El directorio de salida usa el slug del modelo y el esquema aplicado:

```text
out/quantized/
    Modelo-W4A16-AWQ/
        README.md
        config.json
        tokenizer.json
        tokenizer_config.json
        *.safetensors
    Modelo-W4A16-GPTQ/
        README.md
        config.json
        tokenizer.json
        tokenizer_config.json
        *.safetensors
```

Los artefactos se guardan en formato `compressed-tensors`. Cuando dispongas de un entorno `vllm` compatible, puedes validarlos con:

Cada directorio cuantizado incluye ademas un `README.md` en ingles, listo para subir el checkpoint a Hugging Face Hub con el detalle de la corrida: modelo base, dataset de calibracion, esquema, budgets, targets secuenciales y metadata de toolchain.

Antes de escribir un nuevo artefacto, el cuantizador limpia por completo el directorio objetivo del esquema (`...-AWQ` o `...-GPTQ`) para evitar mezclar archivos viejos con los nuevos.

```python
from vllm import LLM

llm = LLM(model="out/quantized/Modelo-W4A16-AWQ", dtype="auto", enforce_eager=True)
```

## Restricciones de VRAM

| Tamaño del modelo | Recomendacion practica                                                                                               |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| 1B a 3B           | Puede calibrarse con amplio margen en 16 GiB.                                                                        |
| 7B                | Requiere `device_map="auto"`, offload a CPU y muestras de calibracion acotadas.                                      |
| >7B               | Requiere validar cuidadosamente offload, batch de calibracion y, en algunos casos, un pipeline secuencial adicional. |

El runner define `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` por defecto si no existe ya en el entorno, para reducir problemas de fragmentacion.

AWQ usa por defecto `128` muestras mediante `QUANTIZER_AWQ_NUM_CALIBRATION_SAMPLES`. Con esto se evita que el smoothing intente retener cientos de salidas intermedias a la vez en una GPU de 16 GiB. Si trabajas con una GPU mayor y quieres volver a usar mas muestras, sube ese valor de forma explicita.

GPTQ usa por defecto un techo mas conservador de `10.0` GiB para el modelo mediante `QUANTIZER_GPTQ_MAX_GPU_MEMORY_GIB`. Ese margen extra deja espacio al Hessian y a la inversion usada durante la cuantizacion de capas grandes como `down_proj`.

Si aparece `OutOfMemoryError`, reduce `QUANTIZER_AWQ_MAX_GPU_MEMORY_GIB` o `QUANTIZER_GPTQ_MAX_GPU_MEMORY_GIB`, baja `QUANTIZER_AWQ_NUM_CALIBRATION_SAMPLES` o `QUANTIZER_GPTQ_NUM_CALIBRATION_SAMPLES`, trabaja con una longitud de secuencia menor o ajusta `QUANTIZER_AWQ_SEQUENTIAL_TARGETS` y `QUANTIZER_GPTQ_SEQUENTIAL_TARGETS`. El valor `safe-auto` prioriza particiones intermedias como `Attention + MLP` cuando la arquitectura las expone. Solo usa `auto` si el modelo ya entra holgadamente en VRAM.

## Excepción de Convención

AGENTS.md recomienda runners raíz `run_*.py`. Este módulo es una herramienta utilitaria autocontenida y por eso su runner vive dentro del propio paquete en `quantizer/run.py`. La excepción está documentada para mantener cohesión y aislar dependencias que hoy no pueden convivir con el entorno principal.
