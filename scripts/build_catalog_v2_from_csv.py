#!/usr/bin/env python3
"""One-off generator: build backend/data/media/catalog_v2.json from the
thesis canonical table combined_geological_media_parameters.csv.

Following the resolution recorded in docs/api_contract_v2_implementation_plan.md
(open question #1, 2026-05-17): the v2 catalog is a JSON projection of
the CSV. Materials without alpha_1_K are flagged
thermoelastic_supported: false. No literature defaults are injected.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = (
    REPO_ROOT.parent
    / "AI_Termoelastic_Waves_Geology/chapters/tables/combined_geological_media_parameters.csv"
)
OUT_PATH = REPO_ROOT / "backend/data/media/catalog_v2.json"


def _float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def main() -> None:
    rows: list[dict] = []
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            mat_id = row["material"].strip()
            if not mat_id:
                continue
            alpha = _float(row.get("alpha_1_K", ""))
            rho = _float(row["rho_kg_m3"])
            vp = _float(row["Vp_m_s"])
            vs = _float(row["Vs_m_s"])
            E = _float(row["E_Pa"])
            nu = _float(row["nu"])
            mu = _float(row["mu_Pa"])
            K = _float(row["K_Pa"])
            lam = _float(row["lambda_Pa"])
            k_th = _float(row["k_W_mK"])
            cp = _float(row["Cp_J_kgK"])
            vol_C = _float(row["C_J_m3K"])
            gamma = _float(row["gamma_Pa_K"])
            phi_str = row.get("porosity_percent", "").strip() or None

            entry = {
                "id": mat_id,
                "name": mat_id.capitalize(),
                "category": row.get("rock_group", "").strip(),
                "thermoelastic_supported": alpha is not None,
                "properties": {
                    "rho_kg_m3": rho,
                    "vp_m_s": vp,
                    "vs_m_s": vs,
                    "young_modulus_pa": E,
                    "poisson_ratio": nu,
                    "shear_modulus_pa": mu,
                    "bulk_modulus_pa": K,
                    "lame_lambda_pa": lam,
                    "thermal_conductivity_w_mk": k_th,
                    "heat_capacity_j_kgk": cp,
                    "volumetric_heat_capacity_j_m3k": vol_C,
                    "thermal_expansion_1_k": alpha,
                    "thermoelastic_gamma_pa_k": gamma,
                    "porosity_summary": phi_str,
                },
                "metadata": {
                    "source_table": "combined_geological_media_parameters.csv",
                    "value_type": row.get("value_type", "mixed"),
                    "source_files": row.get("source_file", ""),
                    "notes": row.get("notes", ""),
                },
            }
            if not entry["thermoelastic_supported"]:
                entry["metadata"]["limitation"] = (
                    "alpha_1_K (thermal expansion) is not available for this "
                    "material in the source table; thermoelastic predictions "
                    "are not supported. Only elastic/geometric quantities are usable."
                )
            rows.append(entry)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    print(f"  total materials: {len(rows)}")
    print(f"  thermoelastic_supported: "
          f"{sum(1 for r in rows if r['thermoelastic_supported'])}")
    print(f"  unsupported (no alpha): "
          f"{sum(1 for r in rows if not r['thermoelastic_supported'])}")


if __name__ == "__main__":
    main()
