# LLMToys

LLMToys es un laboratorio para experimentar con LLMs ejecutados localmente. El proyecto esta pensado para probar modelos, runners, presupuestos de contexto, configuracion de VRAM y flujos completos que combinen pasos deterministas con inferencia local.

El caso de uso mas desarrollado actualmente es **NL2SQL**: una tuberia que recibe una pregunta en lenguaje natural, reduce semanticamente el esquema disponible, compila un plan semantico, genera SQL para un dialecto concreto, valida la salida y, en el flujo orquestado, ejecuta la consulta y redacta una respuesta final.

## Objetivos

- Ejecutar LLMs locales con `vllm` y configuraciones reproducibles.
- Mantener runners simples, configurados por variables de entorno y constantes de modulo.
- Probar distintos modelos para tareas distintas: embeddings, reranking, generacion SQL y narrativa.
- Separar reglas de dominio, prompts y contratos semanticos en YAML versionables o inyectables.
- Validar que las salidas LLM pasen por contratos tipados, normalizacion y guardrails antes de usarse.

## Estructura del repositorio

| Ruta            | Descripcion                                                                                                                                                                      |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `llm_core/`     | Infraestructura comun para cargar y ejecutar modelos locales con vLLM. Incluye registro de modelos, defaults de runtime, utilidades de memoria y conteo/optimizacion de prompts. |
| `nl2sql/`       | Modulo NL2SQL completo. Contiene pruning semantico, resolver semantico, generador SQL y orquestador. Ver [nl2sql/README.md](nl2sql/README.md).                                   |
| `etl/`          | Utilidades locales para inspeccionar una base de datos y exportar esquema/datos a artefactos auxiliares.                                                                         |
| `tests/`        | Suite de pruebas unitarias y de integracion con fixtures genericos.                                                                                                              |
| `run_model.py`  | Runner simple para probar un modelo local registrado en `llm_core`.                                                                                                              |
| `run_nl2sql.py` | Runner orquestado: pruning, resolver, solver, ejecucion SQL y narrativa final.                                                                                                   |

Algunas carpetas de datos/configuracion local, como `schema-docs/`, `out/`, `reports/`, `todos/`, `.cache/` y `.venv/`, estan ignoradas por Git porque pueden contener informacion sensible, artefactos generados o datos de entorno.

## Requisitos

- Linux.
- Python `>=3.12,<3.14.1`.
- `uv` para resolver e instalar dependencias.
- GPU NVIDIA con CUDA compatible si se quieren ejecutar los modelos locales con vLLM.
- Variables de entorno locales para tokens y conexion a base de datos cuando se usen modelos privados o ejecucion SQL.

Las dependencias principales estan declaradas en [pyproject.toml](pyproject.toml): `vllm`, `torch`, `transformers`, `sqlalchemy`, `sqlglot`, `pydantic`, `pyyaml`, `pandas`, `langchain-core` y utilidades relacionadas.

## Instalacion

```bash
uv sync
```

Para activar el entorno virtual local cuando exista:

```bash
source .venv/bin/activate
```

## Configuracion local

La configuracion de runtime se carga desde variables de entorno mediante `python-dotenv`. Los runners no usan `argparse`; cada script define constantes ALL_CAPS como defaults editables y lee overrides desde `.env`.

Variables frecuentes:

| Variable              | Uso                                                               |
| --------------------- | ----------------------------------------------------------------- |
| `HF_TOKEN`            | Token de Hugging Face para modelos gated o privados.              |
| `DATABASE_URL`        | URL SQLAlchemy generica para introspeccion o ejecucion contra BD. |
| `NL2SQL_DATABASE_URL` | URL SQLAlchemy especifica para el orquestador NL2SQL.             |
| `DB_SCHEMA_PATH`      | Ruta al YAML de esquema de base de datos.                         |
| `SEMANTIC_RULES_PATH` | Ruta al contrato semantico YAML.                                  |
| `SQL_DIALECT`         | Dialecto SQL para solver/resolver: `tsql` o `postgres`.           |

No se deben versionar archivos `.env` reales ni outputs generados.

## Uso rapido

Probar un modelo local registrado:

```bash
uv run run_model.py
```

Ejecutar la tuberia NL2SQL por etapas:

```bash
uv run run_semantic_schema_pruning.py
uv run run_semantic_resolver.py
uv run run_sql_solver.py
```

Ejecutar el flujo NL2SQL completo:

```bash
uv run run_nl2sql.py
```

Ejecutar pruebas:

```bash
uv run python -m pytest tests/ -v
```

## Flujo NL2SQL en una linea

```text
pregunta NL -> semantic_prune -> semantic_resolver -> sql_solver_generator -> SQL -> ejecucion -> narrativa
```

Los artefactos intermedios se escriben normalmente en `out/`:

- `out/semantic_pruned_schema.yaml`
- `out/semantic_plan.yaml`
- `out/solver_result.sql`
- `out/solver_result.yaml`
- `out/sql_execution_result.yaml`
- `out/nl2sql_response.yaml`

Para el detalle del modulo, sus YAMLs y sus parametros de configuracion, ver [nl2sql/README.md](nl2sql/README.md).

## Seguridad y publicacion

- Mantener `.env`, dumps, esquemas reales, catalogos reales, outputs y caches fuera de Git.
- Rotar credenciales si alguna vez se copiaron a logs, reportes o conversaciones.
- Tratar `out/` como sensible: puede contener SQL final, previews de filas, rutas de artefactos y detalles del esquema.
- Ejecutar un scanner de secretos antes de publicar el repositorio.
- En entornos compartidos, preferir modelos con revision fija y reducir `trust_remote_code` cuando sea posible.

## Convenciones del proyecto

- Los runners son scripts planos con `main()` y `if __name__ == "__main__": main()`.
- No se usan parsers CLI como `argparse`, `click`, `typer` o similares.
- Las reglas de dominio, prompts y contratos viven en YAML.
- Las etapas principales de NL2SQL se comunican mediante artefactos YAML, no por imports cruzados entre modulos de etapa.
- El codigo generado y los comentarios tecnicos del proyecto se mantienen en espanol cuando aplica.
