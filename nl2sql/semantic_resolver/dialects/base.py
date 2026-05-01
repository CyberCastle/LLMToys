#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Interfaz base para dialectos consumidos por ``semantic_resolver``.

Esta interfaz no pretende cubrir toda la materialización SQL (eso vive en
``sql_solver_generator/dialects``); solo expone las capacidades que el
``plan_compiler`` necesita para emitir un :class:`PlanTimeFilter` y, en
general, cualquier expresión semántica precalculada que dependa del motor
de base de datos.
"""

from __future__ import annotations

from abc import ABC

from ..rules_loader import CompilerRules


class ResolverDialect(ABC):
    """Contrato mínimo de dialecto requerido por el compilador del plan.

    Las implementaciones concretas (``TsqlResolverDialect``,
    ``PostgresResolverDialect``, …) declaran su nombre canónico vía el
    atributo de clase ``name``. Ese nombre se usa como clave en
    ``semantic_resolver.compiler_rules`` (``value_<name>``) y como clave en el
    diccionario ``PlanTimeFilter.resolved_expressions``.
    """

    #: Nombre canónico del dialecto (ej. ``"tsql"``, ``"postgres"``).
    name: str = ""

    def render_time_expression(self, canonical_value: str, *, compiler_rules: CompilerRules | None = None) -> str | None:
        """Convierte una expresión temporal canónica a la sintaxis SQL del dialecto.

        :param canonical_value: Expresión semántica neutra declarada en
            ``semantic_resolver.compiler_rules`` bajo la clave ``value`` (por ejemplo
            ``"today - 1 year"`` o ``"year_start"``).
        :returns: La expresión SQL equivalente lista para inyectar en un
            ``WHERE``, o ``None`` cuando este dialecto no tiene un mapeo de
            respaldo para esa expresión canónica.
        """

        if compiler_rules is None:
            return None
        for time_pattern_rule in compiler_rules.time_patterns:
            if time_pattern_rule.value != canonical_value:
                continue
            resolved_expression = time_pattern_rule.dialect_values.get(self.name)
            if isinstance(resolved_expression, str) and resolved_expression.strip():
                return resolved_expression.strip()
        return None

    def time_pattern_yaml_key(self) -> str:
        """Devuelve la clave esperada en `semantic_resolver.compiler_rules` para este dialecto.

        Por convencion: ``value_<dialect.name>``. Las implementaciones pueden
        sobreescribir este metodo si requieren otra convencion.
        """

        return f"value_{self.name}"
