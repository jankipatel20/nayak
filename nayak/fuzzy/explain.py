"""Explainability utilities for rule inspection and markdown report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from nayak.fuzzy.rulebase import FuzzyRule
from nayak.fuzzy.variables import LinguisticVariable


def format_rule_base_markdown(
    rules: Sequence[FuzzyRule],
    input_variables: Dict[str, LinguisticVariable],
    output_variable: LinguisticVariable,
    min_active_weight: float = 0.10,
    title: str = "NAYAK Fuzzy Rule Base & Membership Report",
) -> str:
    """Format fuzzy rules and membership breakpoints as a clean GitHub-flavored markdown report.

    Args:
        rules: List of FuzzyRule instances.
        input_variables: Dict of input LinguisticVariables.
        output_variable: Output LinguisticVariable.
        min_active_weight: Threshold below which rules are flagged as pruned/inactive.
        title: Report header title.

    Returns:
        Formatted markdown string.
    """
    lines: List[str] = [
        f"# {title}",
        "",
        "## 1. Membership Function Breakpoints (Natural Units)",
        "",
        "| Variable | Term | Type | Breakpoints (Natural Units) |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for var_name, var in input_variables.items():
        if var.crisp_binary:
            lines.append(f"| `{var_name}` | `0` / `1` | Binary Crisp | `{var_name} in [0, 1]` |")
            continue
        for term_name, mf in var.terms.items():
            mf_type = mf.__class__.__name__
            pts = ", ".join(f"{p:.2f}" for p in mf.breakpoints())
            lines.append(f"| `{var_name}` | **{term_name}** | {mf_type} | `({pts})` |")

    for term_name, mf in output_variable.terms.items():
        mf_type = mf.__class__.__name__
        pts = ", ".join(f"{p:.2f}" for p in mf.breakpoints())
        lines.append(f"| `{output_variable.name}` | **{term_name}** | {mf_type} | `({pts})` |")

    lines.extend([
        "",
        "## 2. Active & Pruned Fuzzy Rule Base",
        "",
        f"> Rules with weight $< {min_active_weight:.2f}$ are flagged as effectively pruned/inactive.",
        "",
        "| # | Antecedents (IF) | Consequent (THEN) | Weight | Status |",
        "| :--- | :--- | :--- | :---: | :---: |",
    ])

    for i, r in enumerate(rules, start=1):
        ant_strs = []
        for v, t in r.antecedents.items():
            ant_strs.append(f"`{v}` is **{t}**")
        ant_clause = " AND ".join(ant_strs)
        status = "Active" if r.weight >= min_active_weight else "Pruned (<0.10)"
        lines.append(f"| {i} | {ant_clause} | `extension` is **{r.consequent}** | {r.weight:.3f} | {status} |")

    lines.append("")
    return "\n".join(lines)


def export_rule_report(
    filepath: Path | str,
    rules: Sequence[FuzzyRule],
    input_variables: Dict[str, LinguisticVariable],
    output_variable: LinguisticVariable,
    min_active_weight: float = 0.10,
    title: str = "NAYAK Fuzzy Rule Base & Membership Report",
) -> None:
    """Save rule base report to a markdown file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = format_rule_base_markdown(
        rules=rules,
        input_variables=input_variables,
        output_variable=output_variable,
        min_active_weight=min_active_weight,
        title=title,
    )
    path.write_text(content, encoding="utf-8")
