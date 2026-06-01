"""Apply explainability integration patch to run.py.

Usage:
    python apply_run_explainability_patch.py
    python apply_run_explainability_patch.py /path/to/run.py

This script is intentionally string-based because some generated unified diff
patches are fragile against minor formatting differences in run.py. It only
adds small explainability hooks and keeps the original workflow intact.
"""
from __future__ import annotations

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"[skip] {label}: already patched")
        return text
    if old not in text:
        raise RuntimeError(f"Cannot find patch anchor: {label}")
    return text.replace(old, new, 1)


def insert_after_once(text: str, anchor: str, addition: str, label: str) -> str:
    if addition.strip() in text:
        print(f"[skip] {label}: already patched")
        return text
    if anchor not in text:
        raise RuntimeError(f"Cannot find patch anchor: {label}")
    return text.replace(anchor, anchor + addition, 1)


def patch_run_py(run_path: Path) -> None:
    text = run_path.read_text(encoding="utf-8")
    original = text

    text = insert_after_once(
        text,
        "from factor_mining import (\n"
        "    calculate_factors_performance,\n"
        "    is_low_correlated_with_fixed_factors,\n"
        "    mmr_selection,\n"
        ")\n",
        "from explainability.factor_explainer import explain_factor\n"
        "from explainability.report import save_explainability_artifacts\n",
        "explainability imports",
    )

    text = replace_once(
        text,
        "        profile.setdefault(\"method\", \"factor_mad\")\n"
        "        profile[\"params\"] = profile.get(\"params\") or {}\n"
        "        self.method_profile = profile\n",
        "        profile.setdefault(\"method\", \"factor_mad\")\n"
        "        profile[\"params\"] = profile.get(\"params\") or {}\n"
        "        profile[\"explainability\"] = profile.get(\"explainability\") or {}\n"
        "        self.method_profile = profile\n"
        "        self.explainability_config = profile[\"explainability\"]\n",
        "load_method_config explainability config",
    )

    text = insert_after_once(
        text,
        "        self.logger.info(\n"
        "            f\"挖掘方法详情：method={profile['method']}, params={profile['params']}\"\n"
        "        )\n",
        "\n"
        "    def attach_explainability_to_df(self, factors_df: pd.DataFrame) -> pd.DataFrame:\n"
        "        \"\"\"为候选因子补充 AST、语义标签和解释性评分。\"\"\"\n"
        "        cfg = getattr(self, \"explainability_config\", {}) or {}\n"
        "        if factors_df.empty or not cfg.get(\"enabled\", False):\n"
        "            return factors_df\n"
        "\n"
        "        known_fields = set(getattr(self, \"fields\", []) or [])\n"
        "        ops_info = getattr(self, \"ops_info\", {}) or {}\n"
        "        known_operators = set(ops_info.keys()) if isinstance(ops_info, dict) else set(ops_info)\n"
        "\n"
        "        rows = []\n"
        "        for _, row in factors_df.iterrows():\n"
        "            record = row.to_dict()\n"
        "            expression = str(record.get(\"expression\", \"\"))\n"
        "            explanation = record.get(\"explanation\", \"\")\n"
        "            try:\n"
        "                if pd.isna(explanation):\n"
        "                    explanation = \"\"\n"
        "            except Exception:\n"
        "                pass\n"
        "\n"
        "            try:\n"
        "                exp = explain_factor(\n"
        "                    expression=expression,\n"
        "                    known_fields=known_fields,\n"
        "                    known_operators=known_operators,\n"
        "                    llm_explanation=str(explanation or \"\"),\n"
        "                    config=cfg,\n"
        "                )\n"
        "                record[\"explainability\"] = exp\n"
        "                record[\"interpretability_score\"] = exp.get(\"interpretability_score\", np.nan)\n"
        "                record[\"complexity_penalty\"] = exp.get(\"complexity_penalty\", np.nan)\n"
        "                record[\"semantic_tags\"] = \",\".join(exp.get(\"semantic_tags\", []))\n"
        "                record[\"explainability_parse_status\"] = exp.get(\"parse_status\", \"unknown\")\n"
        "            except Exception as e:\n"
        "                self.logger.warning(f\"因子 {record.get('alpha_id')} 解释性分析失败：{e}\")\n"
        "            rows.append(record)\n"
        "\n"
        "        return pd.DataFrame(rows)\n",
        "attach_explainability_to_df method",
    )

    text = insert_after_once(
        text,
        "        self.logger.info(f\"MMR 筛选（{label}）\")\n",
        "        factors_df = self.attach_explainability_to_df(factors_df)\n",
        "call attach_explainability_to_df",
    )

    text = replace_once(
        text,
        "            lambda_param=self.config[\"mmr_lambda\"],\n"
        "            threshold=self.config[\"mmr_threshold\"],\n"
        "        )\n",
        "            lambda_param=self.config[\"mmr_lambda\"],\n"
        "            threshold=self.config[\"mmr_threshold\"],\n"
        "            explainability_config=getattr(self, \"explainability_config\", {}),\n"
        "        )\n",
        "mmr_selection explainability_config argument",
    )

    text = insert_after_once(
        text,
        "        for alpha_id, fv in final_factor_values.items():\n"
        "            fv.to_parquet(factors_dir / f\"{alpha_id}.pqt\")\n",
        "\n"
        "        exp_cfg = getattr(self, \"explainability_config\", {}) or {}\n"
        "        if exp_cfg.get(\"enabled\", False) and exp_cfg.get(\"report\", {}).get(\"enabled\", False):\n"
        "            default_exp_root = Path(self.config[\"metrics_save_path\"]).parent / \"explainability\"\n"
        "            exp_dir = (\n"
        "                Path(self.config.get(\"explainability_save_path\", default_exp_root))\n"
        "                / f\"cycle_{cycle_count:04d}_{cycle_timestamp}\"\n"
        "            )\n"
        "            for _, row in final_factors_df.iterrows():\n"
        "                exp = row.get(\"explainability\", {})\n"
        "                if not isinstance(exp, dict):\n"
        "                    continue\n"
        "                save_explainability_artifacts(\n"
        "                    factor_id=str(row.get(\"alpha_id\")),\n"
        "                    expression=str(row.get(\"expression\", \"\")),\n"
        "                    explainability=exp,\n"
        "                    metrics=row.to_dict(),\n"
        "                    output_dir=exp_dir,\n"
        "                )\n"
        "            self.logger.info(f\"解释性报告已保存到：{exp_dir}\")\n",
        "save explainability artifacts",
    )

    if text != original:
        backup = run_path.with_suffix(run_path.suffix + ".bak")
        backup.write_text(original, encoding="utf-8")
        run_path.write_text(text, encoding="utf-8")
        print(f"[ok] patched {run_path}")
        print(f"[ok] backup saved to {backup}")
    else:
        print("[ok] no changes needed")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("run.py")
    if not target.exists():
        raise SystemExit(f"run.py not found: {target}")
    patch_run_py(target)
