# NL2SQL

`nl2sql` es el modulo de LLMToys para transformar preguntas en lenguaje natural en consultas SQL validadas. Esta disenado para experimentar con modelos locales en una arquitectura por etapas: cada etapa produce artefactos YAML explicitos y evita acoplarse internamente con las demas.

El flujo objetivo es una pregunta analitica en espanol sobre un esquema relacional multi-tabla. El sistema reduce el esquema visible, resuelve activos semanticos, compila un plan, genera SQL para un dialecto soportado y valida la salida antes de entregarla al orquestador.

## Motivacion del proyecto

Este proyecto nace de una tension practica que aparece en casi cualquier sistema enterprise de text-to-SQL: usar un LLM puro contra un esquema grande da flexibilidad semantica, pero reduce demasiado el control operativo. En bases reales no basta con "generar SQL". Hay que elegir bien tablas, columnas, joins y filtros; limitar el contexto del schema para que escale; respetar dialectos distintos; y evitar que una salida plausible pero incorrecta termine ejecutandose como si fuera confiable.

La evidencia reciente apunta en esa direccion. Benchmarks orientados a flujos enterprise, como Spider 2.0, muestran que el salto entre datasets clasicos y entornos reales sigue siendo grande, sobre todo cuando hay esquemas voluminosos, documentacion dispersa, multiples pasos de razonamiento y diferencias entre motores SQL. La literatura tambien coincide en que el problema central no es solo la sintaxis final, sino el schema linking y la seleccion correcta de relaciones y filtros sobre un contexto muy amplio.

El extremo opuesto tampoco resulta suficiente. Un sistema basado solo en logica determinista, assets curados o templates cerrados puede ser controlable, pero escala mal cuando crecen los dominios, los tipos de preguntas, los dialectos y las variantes del lenguaje natural. En la practica, ese enfoque termina volviendo el sistema rigido y obliga a tocar codigo o reglas especificas con demasiada frecuencia para cubrir nuevas consultas.

La motivacion de `nl2sql` es explorar un punto medio mas razonable: una arquitectura hibrida. En este enfoque, el LLM se usa para interpretar la intencion de la pregunta y proponer filtros, joins y SQL candidatos sobre un contexto compacto; despues, componentes deterministas verifican seguridad, estructura, contrato semantico, compatibilidad con el schema y normalizacion por dialecto antes de permitir ejecucion o respuesta final. La logica determinista no deberia reemplazar al modelo en flexibilidad semantica, sino actuar como capa de control, enforcement y validacion.

Por eso el modulo esta dividido en etapas explicitas. `semantic_prune` reduce el schema visible; `semantic_resolver` transforma activos recuperados en un plan semantico verificable; `sql_solver_generator` genera y normaliza SQL; y el orquestador ejecuta y narra solo despues de pasar por guardrails. El objetivo no es defender una implementacion puramente curada, sino mover el sistema desde generacion acoplada hacia un pipeline donde la flexibilidad del LLM y el control determinista se complementen.

Referencias de contexto:

- Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows. https://proceedings.iclr.cc/paper_files/paper/2025/file/46c10f6c8ea5aa6f267bcdabcb123f97-Paper-Conference.pdf
- Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL. https://arxiv.org/html/2406.08426v8
- RASL: Retrieval Augmented Schema Linking for Massive Database Text-to-SQL. https://assets.amazon.science/1b/95/8f62e89647348f4c4836f6c3040d/rasl-retrieval-augmented-schema-linking-for-massive-database-text-to-sql.pdf
- OWASP Top 10 for Large Language Model Applications. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- SQLGlot. https://github.com/tobymao/sqlglot
- Text-to-SQL solution powered by Amazon Bedrock. https://aws.amazon.com/blogs/machine-learning/text-to-sql-solution-powered-by-amazon-bedrock/

## Limitaciones y supuestos

Este proyecto no debe leerse como una solucion magica ni como un reemplazo completo del modelado semantico del dominio. El pipeline mejora control y auditabilidad frente a un enfoque puramente generativo, pero su comportamiento sigue dependiendo de varios supuestos operativos y de la calidad de los artefactos que alimentan cada etapa.

La limitacion mas importante es que una parte sustancial del funcionamiento correcto depende de un buen desarrollo de `schema-docs/semantic_rules.yaml`. Si las metricas, dimensiones, sinonimos, filtros, relaciones requeridas, invariantes de negocio o ejemplos semanticos estan incompletos, ambiguos o mal modelados, el sistema puede recuperar activos incorrectos, compilar joins insuficientes o producir SQL validado pero semanticamente desalineado con la intencion real del usuario.

Tambien hay una limitacion estructural propia del diseno por etapas: un error temprano no siempre se corrige mas adelante. Si `semantic_prune` elimina tablas o columnas necesarias, el resolver y el solver no pueden reconstruir con fiabilidad contexto que ya se perdio. Del mismo modo, si el plan semantico queda pobre o sesgado, la generacion SQL posterior tiende a heredar esa desviacion.

Otro limite discutido en este trabajo es la rigidez. Aunque el objetivo del modulo es evitar que cada nueva consulta obligue a escribir logica ad hoc en Python, el sistema sigue siendo sensible a gaps de modelado semantico. Cuando aparecen nuevas formas de preguntar, nuevos filtros relevantes o nuevas combinaciones entre entidades, con frecuencia hace falta enriquecer `semantic_rules.yaml`, ajustar assets declarativos o revisar heuristicas configurables para que el pipeline mantenga cobertura sin degradar precision.

Finalmente, este enfoque hibrido implica una disciplina de mantenimiento que no existe en un prototipo "solo prompt". El valor del proyecto aparece cuando schema fisico, reglas semanticas, validaciones y prompts evolucionan de forma coherente. Si esos contratos divergen, el sistema puede volverse mas rigido de lo deseado o dar una falsa sensacion de seguridad por el simple hecho de que la salida final tenga forma valida.

## Flujo funcional

```text
Pregunta NL
  -> semantic_prune
  -> out/semantic_pruned_schema.yaml
  -> semantic_resolver
  -> out/semantic_plan.yaml
  -> sql_solver_generator
  -> out/solver_result.sql + out/solver_result.yaml
  -> orchestrator opcional: ejecucion SQL + narrativa final
```

## Arquitectura del modulo

| Submodulo               | Responsabilidad                                                                                                                                    | Modelos locales principales                                                   | Entrada                                                    | Salida                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------- |
| `semantic_prune/`       | Reducir el esquema fisico a tablas/columnas relevantes. Combina embedding, rerank, MMR, heuristicas estructurales y expansion por FKs.             | `Alibaba-NLP/E2Rank-0.6B`                                                     | Pregunta, `db_schema.yaml`, `semantic_rules.yaml` opcional | `semantic_pruned_schema.yaml`             |
| `semantic_resolver/`    | Recuperar activos semanticos, rerankearlos, resolver sinonimos, construir joins, detectar filtros temporales y compilar un `CompiledSemanticPlan`. | `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B`, verificador opcional | Pregunta, pruned schema, `semantic_rules.yaml`             | `semantic_plan.yaml`                      |
| `sql_solver_generator/` | Convertir el plan semantico en SQL final, normalizarlo con `sqlglot` y aplicar validaciones AST/dialecto/reglas.                                   | `XGenerationLab/XiYanSQL-QwenCoder-7B-2504`                                   | Semantic plan, pruned schema, reglas semanticas            | `solver_result.sql`, `solver_result.yaml` |
| `orchestrator/`         | Encadenar las etapas, ejecutar SQL con SQLAlchemy y redactar narrativa final.                                                                      | Runner narrativo configurado en `llm_core`                                    | `NL2SQLRequest`                                            | `NL2SQLResponse`, artefactos en `out/`    |
| `utils/`                | Contratos compartidos, normalizacion YAML, modelos de decision, cache de embeddings y utilidades textuales.                                        | No aplica                                                                     | Objetos internos                                           | Utilidades reutilizables                  |

## Requerimientos

- Python `>=3.12,<3.14.1`.
- Dependencias instaladas con `uv sync`.
- GPU NVIDIA/CUDA para ejecucion local de modelos con vLLM.
- `HF_TOKEN` si algun modelo remoto lo requiere.
- Esquema de BD serializado en YAML o acceso a `DATABASE_URL` para reflection.
- Contrato semantico del dominio en YAML.

Para ejecutar contra una base real, el orquestador necesita una URL SQLAlchemy mediante `NL2SQL_DATABASE_URL`, `DATABASE_URL`, `NL2SQL_TSQL_URL` o `NL2SQL_POSTGRES_URL`, segun el dialecto.

## Ejecucion por etapas

```bash
uv run run_semantic_schema_pruning.py
uv run run_semantic_resolver.py
uv run run_sql_solver.py
```

## Ejecucion orquestada

```bash
uv run run_nl2sql.py
```

El runner orquestado construye un `NL2SQLRequest` con pregunta, paths de esquema/reglas/catalogos, directorio de salida y dialecto. Luego ejecuta la secuencia definida en `orchestrator/pipeline.py`:

```text
init -> prune -> resolver -> solver -> execution -> narrative -> response
```

## YAMLs requeridos por datos de dominio

### `schema-docs/db_schema.yaml`

Describe el esquema fisico disponible para retrieval y validacion. Puede generarse desde reflection con `etl/inspect_db.py` o mantenerse manualmente.

Estructura esperada:

```yaml
tabla_base:
    description: Descripcion funcional de la tabla
    columns:
        - name: id
          type: BIGINT
          description: Identificador principal
        - name: created_at
          type: DATE
    primary_keys:
        - id
    foreign_keys:
        - col: otra_tabla_id
          ref_table: otra_tabla
          ref_col: id
```

Notas:

- `columns` tambien puede normalizarse desde mappings internos, pero el formato de lista con `name` y `type` es el mas claro para versionar.
- `description` y `column_descriptions` ayudan al retrieval semantico.
- `foreign_keys` es critico para expansion estructural, join paths y validacion de rutas.

### `schema-docs/semantic_rules.yaml`

Es el contrato semantico principal del dominio. Para que sea portable entre negocios, el archivo fisico se organiza en secciones top-level con anchors YAML y termina con una raiz `semantic_contract` que referencia solo esas secciones activas.

Orden recomendado del archivo:

1. `ACTIVOS SEMANTICOS RECUPERABLES`: entidades, dimensiones, metricas, filtros, relaciones y restricciones que alimentan embeddings/rerank, scoring y compilacion del plan.
2. `RUTAS Y METRICAS DERIVADAS DE LOGICA NL2SQL`: `semantic_join_paths` y `semantic_derived_metrics`, consumidas por prune/resolver/solver.
3. `HEURISTICAS DE RETRIEVAL Y EJEMPLOS PARA LLMS`: sinonimos y ejemplos compactables para verifier/solver.
4. `REGLAS DETERMINISTAS Y GUARDRAILS SQL`: reglas del compilador, senales de pruning y seguridad SQL.
5. `CONTRATO CANONICO`: `semantic_contract`, unico mapa estable consumido por el modulo.

No agregar bloques top-level fuera de ese contrato salvo que exista un loader explicito en `nl2sql`. Las secciones historicas `*_extended`, `semantic_intents`, `semantic_examples_with_intent` y `semantic_source_corrections` fueron eliminadas porque no eran consumidas por la logica ni por los LLMs.

Estructura base:

```yaml
semantic_models: &semantic_models
    - name: modelo_operativo
      core_tables: []

semantic_entities: &semantic_entities
    - name: entity_a
      source_table: entity_a
      key: entity_a.id

# ... definir el resto de anchors activos ...

semantic_contract:
    business_invariants:
        semantic_models: *semantic_models
        semantic_entities: *semantic_entities
        semantic_dimensions: *semantic_dimensions
        semantic_metrics: *semantic_metrics
        semantic_filters: *semantic_filters
        semantic_business_rules: *semantic_business_rules
        semantic_relationships: *semantic_relationships
        semantic_constraints: *semantic_constraints
        semantic_join_paths: *semantic_join_paths
        semantic_derived_metrics: *semantic_derived_metrics
    retrieval_heuristics:
        semantic_synonyms: *semantic_synonyms
        semantic_examples: *semantic_examples
    sql_safety:
        execution_safety: *execution_safety
        hard_join_blacklist: *hard_join_blacklist
        semantic_sql_business_rules: *semantic_sql_business_rules
```

Campos removidos/no soportados en la plantilla actual: `identifiers`, `related_tables`, `calculation_logic`, `dependencies`, `source_values`, `required_tables`, `filters_allowed`, `post_aggregation_params`, `sql_sketch`, `source_file` y metadatos `expected_*` dentro de ejemplos anidados de metricas. Si alguno vuelve a ser necesario, primero debe agregarse soporte explicito en el loader o en los payloads compactos enviados a los LLMs.

#### `business_invariants.semantic_models`

Agrupa tablas que forman un modelo semantico coherente.

```yaml
- name: modelo_operativo
  core_tables:
      - fact_table
      - dimension_table
  grain:
      - fact_table
```

Campos habituales:

- `name`: identificador del modelo.
- `core_tables`: tablas principales disponibles para ese modelo.
- `grain`: tablas o entidades que definen el grano preferido.

#### `business_invariants.semantic_entities`

Mapea entidades de negocio a tablas fisicas.

```yaml
- name: entity_a
  business_definition: Registro operacional base.
  source_table: entity_a
  key: entity_a.id
  time_field: entity_a.created_at
```

Campos habituales:

- `name`: nombre semantico.
- `business_definition`: descripcion funcional.
- `source_table`: tabla fisica.
- `key`: clave principal calificada.
- `time_field`: campo temporal por defecto para filtros relativos.

No usar `identifiers` ni `related_tables`: no son leidos por `nl2sql`. Los identificadores de negocio deben modelarse como dimensiones/filtros, y las relaciones deben vivir en `semantic_relationships` o `semantic_join_paths`.

#### `business_invariants.semantic_dimensions`

Define dimensiones seleccionables o agrupables.

```yaml
- name: entity_label
  entity: entity_a
  source: entity_a.display_name
  type: string
```

Campos habituales:

- `name`: identificador de dimension.
- `entity`: entidad semantica asociada.
- `source`: columna fisica calificada `tabla.columna`.
- `type`: `string`, `date`, `datetime`, `numeric`, `id`, etc.

#### `business_invariants.semantic_metrics`

Declara metricas computables y sus dependencias.

```yaml
- name: metric_count_a
  entity: entity_a
  formula: count_distinct(entity_a.id)
  synonyms:
      - cantidad de registros
  source_catalog:
      table: status_table
      key_column: id
      value_column: name
  required_relationships:
      - from: entity_a.status_id
        to: status_table.id
```

Campos habituales:

- `name`: nombre canonico de la metrica.
- `entity`: entidad base.
- `formula`: expresion semantica o SQL-like con referencias `tabla.columna`.
- `synonyms`: frases alternativas para retrieval.
- `source_catalog`: catalogo necesario para interpretar filtros de estado o clasificacion.
- `required_relationships`: FKs que deben preservarse aunque el top-k no las recupere.
- `examples`: ejemplos anidados opcionales con `question` y `expected_metric` para reforzar matching exacto de metricas.

No usar `source_values`, `calculation_logic`, `dependencies` ni `required_tables`: esos campos no participan en pruning, resolver, solver ni prompts. Si la metrica requiere preservar estructura, usar `formula`, `source_catalog`, `required_relationships` o `tables`.

#### `business_invariants.semantic_filters`

Declara filtros semanticos reutilizables.

```yaml
- name: by_entity
  field: entity_a.id
```

#### `business_invariants.semantic_relationships`

Declara relaciones semanticas entre columnas.

```yaml
- from: entity_a.parent_id
  to: entity_parent.id
```

Se usan para construir el grafo de joins cuando el esquema fisico no basta o cuando se quiere reforzar una relacion semantica.

#### `business_invariants.semantic_join_paths`

Define rutas canonicas de join que deben tener mas autoridad que un BFS por FKs.

```yaml
- name: entity_to_group_path
  from_entity: entity_group
  to_entity: entity_a
  path:
      - entity_group.id = bridge.group_id
      - bridge.entity_a_id = entity_a.id
```

Campos habituales:

- `name`: identificador de la ruta.
- `from_entity`: entidad origen.
- `to_entity`: entidad destino.
- `path`: lista ordenada de igualdades `tabla.col = tabla.col`.

#### `business_invariants.semantic_derived_metrics`

Describe metricas de dos niveles, por ejemplo promedios sobre conteos agrupados.

```yaml
- name: metric_avg_a_per_group
  base_measure: metric_count_a
  base_group_by:
      - entity_group.id
  post_aggregation: avg
  join_path_hint: entity_to_group_path
```

Campos habituales:

- `base_measure`: metrica base declarada en `semantic_metrics`.
- `base_group_by`: agrupacion interna obligatoria.
- `post_aggregation`: agregacion externa (`avg`, `sum`, `count`, etc.).
- `join_path_hint`: ruta canonica recomendada.

No usar `filters_allowed`, `post_aggregation_params` ni `sql_sketch`: el loader actual solo consume `name`, `description`, `base_measure`, `base_group_by`, `post_aggregation`, `join_path_hint` y `synonyms`.

#### `retrieval_heuristics.semantic_synonyms`

Mapping de entidad a sinonimos.

```yaml
semantic_synonyms:
    entity_a:
        - EA
        - registro operativo
```

#### `retrieval_heuristics.semantic_examples`

Ejemplos curados que ayudan a inyectar metricas/dimensiones/modelos cuando una pregunta parecida entra al top-k.

```yaml
- question: cual es el promedio de registros activos por entidad?
  model: modelo_operativo
  metrics:
      - metric_count_a_active
  dimensions:
      - entity_label
```

La configuracion tecnica del pipeline ya no forma parte de `semantic_contract`. Prompts, heuristicas y reglas operativas viven en `nl2sql/config/settings.yaml` para evitar mezclar contrato semantico de dominio con tuning interno del runtime.

#### `sql_safety.execution_safety`

Lista keywords prohibidas y restricciones generales de ejecucion.

```yaml
execution_safety:
    forbidden_keywords:
        - DROP
        - DELETE
        - UPDATE
```

#### `sql_safety.hard_join_blacklist`

Lista tablas que no deben usarse como join fuerte sin regla declarativa.

```yaml
hard_join_blacklist:
    - table: external_table
      reason: homologacion declarada
```

#### `sql_safety.semantic_sql_business_rules`

Reglas declarativas consumidas por `sql_solver_generator.business_rules`.

Tipos soportados actualmente:

- `inject_filter_if_column_present`
- `forbid_column`
- `require_filter_when_table_used`
- `require_table_when_table_used`
- `require_normalization_before_sum`

Ejemplo:

```yaml
semantic_sql_business_rules:
    - id: forbid_sensitive_column
      type: forbid_column
      column: table_name.sensitive_column
    - id: require_scope_filter
      type: require_filter_when_table_used
      table: table_name
      any_of:
          - field: table_name.scope_id
            operator: =
            value: 1
```

### `schema-docs/catalogos-embebidos.yml`

Archivo reservado para catalogos embebidos o valores de referencia que pueden complementar reglas semanticas y filtros. El runner orquestado acepta la ruta como `catalogos_path` dentro de `NL2SQLRequest`.

Estructura recomendada:

```yaml
catalogs:
    status_table:
        key_column: id
        value_column: name
        values:
            - id: 1
              name: Active
            - id: 2
              name: Archived
```

Nota: en el estado actual del codigo, este path queda modelado en el contrato del orquestador y puede usarse como extension declarativa; las reglas semanticas principales siguen viviendo en `semantic_rules.yaml`.

## YAML interno configurable

Toda la configuracion interna de `nl2sql` vive ahora en `nl2sql/config/settings.yaml`.

Secciones principales:

- `semantic_prune.prompts`
- `semantic_prune.heuristic_rules`
- `semantic_prune.query_signal_rules`
- `semantic_resolver.prompts`
- `semantic_resolver.compiler_rules`
- `sql_solver.prompts`
- `sql_solver.filter_value_rules`
- `orchestrator.narrative_prompt`

`NL2SQL_CONFIG_PATH` es la unica variable soportada para redirigir ese archivo completo. Los assets internos de prune, resolver, solver y narrativa ya no aceptan overrides por etapa mediante variables `*_PATH`.

## Artefactos YAML generados

### `out/semantic_pruned_schema.yaml`

Producido por `semantic_prune`.

Estructura simplificada:

```yaml
query: pregunta original
retrieval_query: pregunta enriquecida opcional
pruned_schema:
    table_name:
        columns:
            - name: id
              type: BIGINT
        primary_keys: [id]
        foreign_keys: []
        selection_reason: semantic_join_path
```

### `out/semantic_plan.yaml`

Producido por `semantic_resolver`.

Estructura simplificada:

```yaml
semantic_plan:
    retrieved_candidates:
        query: pregunta
        assets_by_kind: {}
        diagnostics: {}
    compiled_plan:
        query: pregunta
        semantic_model: model_name
        intent: post_aggregated_metric
        base_entity: entity_a
        grain: entity_a.id
        measure:
            name: metric_count_a
            formula: count_distinct(entity_a.id)
            source_table: entity_a
        group_by:
            - entity_c.id
        time_filter:
            field: entity_a.created_at
            operator: ">="
            value: today - 1 year
        join_path: []
        required_tables: []
        verification:
            is_semantically_aligned: true
```

### `out/solver_result.yaml`

Producido por `sql_solver_generator`.

Estructura simplificada:

```yaml
sql_final: SELECT ...
sql_query_spec:
    query_type: derived_metric
    dialect: tsql
    base_entity: entity_a
    base_table: entity_a
    selected_metrics: []
    join_plan: []
metadata:
    tables_used: []
    columns_used: []
    join_paths_used: []
    model_used: XGenerationLab/XiYanSQL-QwenCoder-7B-2504
warnings: []
issues: []
```

### `out/sql_execution_result.yaml`

Producido por el orquestador cuando ejecuta SQL.

```yaml
query: pregunta original
sql: SELECT ...
row_count: 10
truncated: false
execution_seconds: 0.12
rows:
    - column: value
```

### `out/nl2sql_response.yaml`

Respuesta consolidada del runner orquestado.

```yaml
query: pregunta original
final_sql: SELECT ...
rows: []
row_count: 0
truncated: false
narrative: respuesta final
artifacts: []
warnings: []
issues: []
```

## Parametros de configuracion

### Runners y rutas generales

| Variable                      | Default                                                          | Descripcion                                           |
| ----------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------- |
| `SEMANTIC_QUERY`              | Pregunta de ejemplo del runner                                   | Pregunta en lenguaje natural para runners por etapa.  |
| `DB_SCHEMA_PATH`              | `schema-docs/db_schema.yaml` o `out/db_schema.yaml` segun runner | Esquema fisico de BD.                                 |
| `SEMANTIC_RULES_PATH`         | `schema-docs/semantic_rules.yaml`                                | Contrato semantico del dominio.                       |
| `SEMANTIC_PRUNED_SCHEMA_PATH` | `out/semantic_pruned_schema.yaml`                                | Artefacto entre prune y resolver/solver.              |
| `SEMANTIC_PLAN_PATH`          | `out/semantic_plan.yaml`                                         | Artefacto entre resolver y solver.                    |
| `SQL_DIALECT`                 | `tsql`                                                           | Dialecto para resolver/solver por etapas.             |
| `NL2SQL_DIALECT`              | `tsql`                                                           | Dialecto del runner orquestado.                       |
| `NL2SQL_OUT_DIR`              | `out`                                                            | Directorio de artefactos del orquestador.             |
| `NL2SQL_RESPONSE_PATH`        | `out/nl2sql_response.yaml`                                       | YAML consolidado del orquestador.                     |
| `CATALOGOS_PATH`              | `schema-docs/catalogos-embebidos.yml`                            | Ruta de catalogos embebidos opcionales en el request. |

### Conexiones y ejecucion SQL

| Variable                    | Descripcion                                                           |
| --------------------------- | --------------------------------------------------------------------- |
| `DATABASE_URL`              | URL SQLAlchemy generica usada por reflection y fallback de ejecucion. |
| `NL2SQL_DATABASE_URL`       | URL SQLAlchemy preferida por el orquestador.                          |
| `NL2SQL_TSQL_URL`           | URL especifica cuando el dialecto es `tsql`.                          |
| `NL2SQL_POSTGRES_URL`       | URL especifica cuando el dialecto es `postgres`.                      |
| `NL2SQL_MAX_ROWS`           | Maximo de filas a recuperar en ejecucion SQL. Default: `1000`.        |
| `NL2SQL_ROWS_PREVIEW_LIMIT` | Filas incluidas en prompt/reporte narrativo. Default: `25`.           |

### `semantic_prune`

| Variable                                      | Default                    | Descripcion                                        |
| --------------------------------------------- | -------------------------- | -------------------------------------------------- |
| `SEMANTIC_PRUNE_MODEL`                        | `Alibaba-NLP/E2Rank-0.6B`  | Modelo para embedding/rerank de esquema.           |
| `SEMANTIC_PRUNE_DTYPE`                        | `auto`                     | Precision del runtime.                             |
| `SEMANTIC_PRUNE_MAX_MODEL_LEN`                | `30464`                    | Contexto maximo del modelo.                        |
| `SEMANTIC_PRUNE_GPU_MEMORY_UTILIZATION`       | `0.30`                     | Fraccion de VRAM para vLLM.                        |
| `SEMANTIC_PRUNE_TENSOR_PARALLEL_SIZE`         | `1`                        | Paralelismo tensorial.                             |
| `SEMANTIC_PRUNE_CACHE_DIR`                    | `.cache/schema_embeddings` | Cache persistente de embeddings.                   |
| `SEMANTIC_PRUNE_ENABLE_QUERY_ENRICHMENT`      | `true`                     | Enriquecer la pregunta con senales.                |
| `SEMANTIC_PRUNE_LISTWISE_USES_ENRICHED_QUERY` | `false`                    | Usar query enriquecida tambien en listwise rerank. |
| `SEMANTIC_PRUNE_ENABLE_EMBEDDING_CACHE`       | `true`                     | Reutilizar embeddings de esquema.                  |
| `SEMANTIC_PRUNE_TOP_K_MATCHES`                | `12`                       | Candidatos globales.                               |
| `SEMANTIC_PRUNE_TOP_K_TABLES`                 | `5`                        | Tablas a retener por ranking.                      |
| `SEMANTIC_PRUNE_TOP_K_COLUMNS_PER_TABLE`      | `6`                        | Columnas por tabla.                                |
| `SEMANTIC_PRUNE_MIN_SCORE`                    | `0.20`                     | Score minimo semantico.                            |
| `SEMANTIC_PRUNE_SHOW_MATCH_TEXT`              | `false`                    | Mostrar texto de matches en reporte.               |
| `SEMANTIC_PRUNE_SHOW_PRUNED_SCHEMA`           | `true`                     | Mostrar schema podado en reporte.                  |

Los assets internos de `semantic_prune` se toman siempre desde el YAML unificado. Para usar un documento distinto, redirige `NL2SQL_CONFIG_PATH`.

### `semantic_resolver`

| Variable                                                 | Default                               | Descripcion                                                    |
| -------------------------------------------------------- | ------------------------------------- | -------------------------------------------------------------- |
| `SEMANTIC_RESOLVER_EMBEDDING_MODEL`                      | `Qwen/Qwen3-Embedding-0.6B`           | Modelo de embeddings de activos semanticos.                    |
| `SEMANTIC_RESOLVER_RERANKER_MODEL`                       | `Qwen/Qwen3-Reranker-0.6B`            | Modelo reranker yes/no.                                        |
| `SEMANTIC_RESOLVER_DTYPE`                                | `auto`                                | Precision del runtime.                                         |
| `SEMANTIC_RESOLVER_MAX_MODEL_LEN`                        | `8192`                                | Contexto maximo embed/rerank.                                  |
| `SEMANTIC_RESOLVER_GPU_MEMORY_UTILIZATION_EMBED`         | `0.30`                                | VRAM para embedding.                                           |
| `SEMANTIC_RESOLVER_GPU_MEMORY_UTILIZATION_RERANK`        | `0.25`                                | VRAM para rerank.                                              |
| `SEMANTIC_RESOLVER_TENSOR_PARALLEL_SIZE`                 | `1`                                   | Paralelismo tensorial.                                         |
| `SEMANTIC_RESOLVER_TRUST_REMOTE_CODE`                    | `true`                                | Permitir codigo remoto del modelo.                             |
| `SEMANTIC_RESOLVER_TOP_K_RETRIEVAL`                      | `40`                                  | Candidatos recuperados por embedding.                          |
| `SEMANTIC_RESOLVER_TOP_K_RERANK`                         | `40`                                  | Candidatos rerankeados.                                        |
| `SEMANTIC_RESOLVER_MIN_EMBEDDING_SCORE`                  | `0.25`                                | Umbral embedding.                                              |
| `SEMANTIC_RESOLVER_MIN_RERANK_SCORE`                     | `0.0`                                 | Umbral rerank.                                                 |
| `SEMANTIC_RESOLVER_COMPATIBILITY_MIN_SCORE`              | `0.20`                                | Umbral de compatibilidad semantica.                            |
| `SEMANTIC_RESOLVER_ENABLE_SYNONYM_QUERY_EXPANSION`       | `true`                                | Expandir query por sinonimos.                                  |
| `SEMANTIC_RESOLVER_ENABLE_SYNONYM_SCORE_BOOST`           | `true`                                | Aplicar boost por sinonimos.                                   |
| `SEMANTIC_RESOLVER_SYNONYM_QUERY_EXPANSION_MAX_ENTITIES` | `6`                                   | Maximo de entidades expandidas.                                |
| `SEMANTIC_RESOLVER_SYNONYM_DIRECT_BOOST`                 | `0.20`                                | Boost directo.                                                 |
| `SEMANTIC_RESOLVER_SYNONYM_RELATED_BOOST`                | `0.08`                                | Boost relacionado.                                             |
| `SEMANTIC_RESOLVER_RERANK_BATCH_SIZE`                    | `8`                                   | Batch de rerank.                                               |
| `SEMANTIC_RESOLVER_RERANK_MAX_DOCUMENT_CHARS`            | `2400`                                | Tamano maximo del documento rerankeado.                        |
| `SEMANTIC_RESOLVER_RERANK_LOGPROBS`                      | `20`                                  | Logprobs para scoring yes/no.                                  |
| `SEMANTIC_RESOLVER_RERANK_PROMPT_TOKEN_MARGIN`           | `16`                                  | Margen de tokens en prompt rerank.                             |
| `SEMANTIC_RESOLVER_SEQUENTIAL_ENGINES`                   | `true`                                | Cargar embedding y reranker secuencialmente para ahorrar VRAM. |
| `SEMANTIC_RESOLVER_SHOW_REJECTED_ASSETS`                 | `true`                                | Mostrar candidatos rechazados.                                 |
| `SEMANTIC_RESOLVER_ENABLE_PLAN_COMPILER`                 | `true`                                | Compilar `CompiledSemanticPlan`.                               |
| `SEMANTIC_RESOLVER_DEFAULT_POST_AGGREGATION_FUNCTION`    | `avg`                                 | Agregacion post grupo por defecto.                             |
| `SEMANTIC_RESOLVER_EMBEDDING_CACHE_DIR`                  | `.cache/semantic_resolver_embeddings` | Cache de embeddings.                                           |

Los prompts y reglas internas de `semantic_resolver` se cargan desde el YAML unificado. Usa `NL2SQL_CONFIG_PATH` si necesitas apuntar a otro documento completo.

#### Verificador semantico

| Variable                                            | Default                              | Descripcion                                                                       |
| --------------------------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------- |
| `SEMANTIC_RESOLVER_ENABLE_SEMANTIC_VERIFIER`        | `true`                               | Activar verificacion LLM del plan.                                                |
| `SEMANTIC_RESOLVER_VERIFIER_MODEL`                  | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` | Modelo verificador.                                                               |
| `SEMANTIC_RESOLVER_VERIFIER_DTYPE`                  | `auto`                               | Precision del verificador.                                                        |
| `SEMANTIC_RESOLVER_VERIFIER_TEMPERATURE`            | `0.0`                                | Temperatura del verificador.                                                      |
| `SEMANTIC_RESOLVER_VERIFIER_MAX_MODEL_LEN`          | `2048`                               | Contexto maximo.                                                                  |
| `SEMANTIC_RESOLVER_VERIFIER_MAX_TOKENS`             | `256`                                | Tokens de salida.                                                                 |
| `SEMANTIC_RESOLVER_VERIFIER_GPU_MEMORY_UTILIZATION` | `0.82`                               | VRAM del verificador.                                                             |
| `SEMANTIC_RESOLVER_VERIFIER_CPU_OFFLOAD_GB`         | `0.0`                                | Offload CPU.                                                                      |
| `SEMANTIC_RESOLVER_VERIFIER_ENFORCE_EAGER`          | `true`                               | Forzar modo eager del verificador; `true` desactiva `torch.compile` y cudagraphs. |
| `SEMANTIC_RESOLVER_VERIFIER_FEW_SHOT_LIMIT`         | `3`                                  | Ejemplos curados maximos en prompt.                                               |

### `sql_solver_generator`

| Variable                              | Default                                     | Descripcion                                                                                                                          |
| ------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `SQL_DIALECT`                         | `tsql`                                      | Dialecto de SQL final: `tsql` o `postgres`.                                                                                          |
| `SQL_SOLVER_MODEL`                    | `XGenerationLab/XiYanSQL-QwenCoder-7B-2504` | Modelo generador SQL. Si quiere usar un checkpoint cuantizado local, indique aqui la ruta del modelo.                                |
| `SQL_SOLVER_MAX_RETRIES`              | `1`                                         | Reintentos con reglas de reparacion.                                                                                                 |
| `SQL_SOLVER_LLM_DTYPE`                | `bfloat16`                                  | Precision del modelo.                                                                                                                |
| `SQL_SOLVER_MAX_MODEL_LEN`            | `2048`                                      | Contexto maximo.                                                                                                                     |
| `SQL_SOLVER_MAX_TOKENS`               | `384`                                       | Tokens de salida.                                                                                                                    |
| `SQL_SOLVER_TEMPERATURE`              | `0.0`                                       | Temperatura de generacion.                                                                                                           |
| `SQL_SOLVER_GPU_MEMORY_UTILIZATION`   | `0.90`                                      | VRAM para vLLM.                                                                                                                      |
| `SQL_SOLVER_ENFORCE_EAGER`            | `true`                                      | Modo eager para reducir fallos de memoria.                                                                                           |
| `SQL_SOLVER_CPU_OFFLOAD_GB`           | `3.0`                                       | Offload CPU.                                                                                                                         |
| `SQL_SOLVER_MIN_CPU_OFFLOAD_GB`       | `3.0`                                       | Piso minimo de offload CPU aplicado por el runtime del solver. Puede bajarse a `0.0` para checkpoints cuantizados si la GPU alcanza. |
| `SQL_SOLVER_SWAP_SPACE_GB`            | `4.0`                                       | Swap space vLLM.                                                                                                                     |
| `SQL_SOLVER_FAIL_ON_VALIDATION_ERROR` | `false`                                     | Si `true`, aborta ante issues de validacion. Recomendado para ejecucion contra BD real.                                              |

Los prompts y reglas lexicas de `sql_solver_generator` tambien salen del YAML unificado; no hay overrides dedicados por variable de entorno.

Cuando se ejecuta `run_sql_solver.py` o `run_nl2sql.py`, `SQL_SOLVER_MODEL` usa por defecto el repo HF canonico del solver. Si se redefine con una ruta existente, el runner la normaliza a path absoluto antes de instanciar el solver, por ejemplo `out/quantized/XiYanSQL-QwenCoder-7B-2504-W4A16-AWQ`.

### Narrativa del orquestador

| Variable                    | Default | Descripcion                         |
| --------------------------- | ------- | ----------------------------------- |
| `NL2SQL_MAX_ROWS`           | `1000`  | Filas maximas a recuperar.          |
| `NL2SQL_ROWS_PREVIEW_LIMIT` | `25`    | Filas maximas en preview narrativo. |

El prompt narrativo tambien se toma desde el YAML unificado; no existe override dedicado por variable de entorno.

## Contratos internos relevantes

### `CompiledSemanticPlan`

El resolver compila una estructura con:

- `query`
- `semantic_model`
- `intent`
- `base_entity`
- `grain`
- `measure`
- `group_by`
- `time_filter`
- `post_aggregation`
- `join_path`
- `required_tables`
- `join_path_hint`
- `derived_metric_ref`
- `population_scope`
- `base_group_by`
- `intermediate_alias`
- `verification`
- `issues`

### `SQLQuerySpec`

El solver traduce el plan a un contrato SQL con:

- `query_type`: `scalar_metric`, `grouped_metric`, `derived_metric`, `ranking` o `detail_listing`.
- `dialect`: `tsql` o `postgres`.
- `base_entity`, `base_table`.
- `selected_metrics`, `selected_dimensions`, `selected_filters`.
- `time_filter`.
- `join_plan`.
- `base_group_by`, `final_group_by`.
- `post_aggregation`.
- `limit`.
- `warnings`.

El LLM generador no debe emitir este contrato completo; debe emitir solamente JSON con `final_sql`. El contrato se construye y valida dentro del pipeline.

## Guardrails y validacion

El solver aplica varias capas de control:

- Prohibicion lexica de keywords definidas en `execution_safety.forbidden_keywords`.
- Parseo y normalizacion con `sqlglot`.
- Rechazo de DML/DDL, multiples sentencias, `CROSS JOIN` y joins sin `ON`.
- Validacion de columnas desnudas desconocidas.
- Validacion de filtros `WHERE` no declarados.
- Validacion de forma para metricas derivadas de dos niveles.
- Reglas declarativas de negocio desde `semantic_sql_business_rules`.

Para ejecutar contra una base real, se recomienda activar:

```bash
SQL_SOLVER_FAIL_ON_VALIDATION_ERROR=true
```

## Consideraciones de seguridad

- No versionar `.env`, esquemas reales, catalogos reales, dumps ni `out/`.
- Tratar `semantic_plan.yaml`, `solver_result.yaml`, `solver_result.sql` y `sql_execution_result.yaml` como artefactos sensibles.
- Usar credenciales de solo lectura para ejecucion SQL.
- Limitar `NL2SQL_MAX_ROWS` y revisar `rows_preview_limit` antes de compartir respuestas.
- Rotar tokens/credenciales si se copiaron a reportes o logs.

## Extension del modulo

Para agregar un nuevo dominio, normalmente basta con:

1. Crear `db_schema.yaml` con tablas, columnas, PKs y FKs.
2. Crear `semantic_rules.yaml` con entidades, metricas, dimensiones, relaciones y reglas SQL.
3. Ajustar prompts o reglas internas con YAMLs alternativos si el lenguaje del dominio lo requiere.
4. Ejecutar las etapas por separado y revisar cada artefacto antes de usar el orquestador completo.

Para agregar un nuevo dialecto SQL, crear una implementacion en `semantic_resolver/dialects/` y `sql_solver_generator/dialects/`, registrarla en sus registries y agregar tests de materializacion temporal, normalizacion y limites de filas.
