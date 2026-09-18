#!/usr/bin/env python3
"""Generate docs/연구노트/experiment_index.csv from the original result files and the plan manifest.

Why this exists: 업무 지침 §10 forbids hand-filling performance numbers into the index
("CSV의 성능 숫자를 수기로 채우거나 계획에 가짜 results.json을 만들지 않는다"), and §11
forbids back-filling unknown provenance with the current environment's values
("모르는 값을 false나 현재 환경 값으로 소급 입력하지 않는다").

So the index is generated, never edited by hand:

  completed runs -> extracted from results/**/results.json (the original artifact)
  planned/blocked -> read from experiment_plan.yaml, result fields left empty
  missing provenance -> the literal token `no_record`, never `false` and never a guess

ID scheme. The historical runs predate any experiment-ID convention, so they get a stable
`EXP-LEGACY-<slug>` derived from their result directory rather than a fabricated date.
New experiments use the §11 form `EXP-YYYYMMDD-NNN` and live in the plan manifest.

Usage:
    python3 docs/연구노트/build_experiment_index.py            # write the CSV
    python3 docs/연구노트/build_experiment_index.py --check    # verify it is up to date
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOC_DIR.parents[1]
OUT_CSV = DOC_DIR / "experiment_index.csv"
PLAN_YAML = DOC_DIR / "experiment_plan.yaml"

NO_RECORD = "no_record"
NA = "not_applicable"

COLUMNS = [
    "experiment_id", "hypothesis_id", "status", "purpose",
    "data_version", "raw_manifest_sha256", "annotation_version",
    "split_manifest_sha256", "quality_policy_version", "input_schema_version",
    "code_commit", "git_dirty", "environment_profile", "environment_record",
    "config_path", "seeds", "seed_controls_init",
    "primary_metric", "primary_metric_value", "secondary_metrics",
    "comparison", "stopping_rule", "result_location", "limitations", "decision",
]

# Historical runs: result dir -> (id slug, hypothesis, purpose).
# The hypothesis mapping follows decisions.md D-002 full IDs.
LEGACY = {
    "results/baseline": (
        "baseline-v1", "AIHUB-H1",
        "V1 Multimodal LSTM baseline. 공급자 공식 아키텍처와 동일 파라미터 수"),
    "results/ablation_v2": ("ablation-v2", "AIHUB-H1", "Ablation V2 (multiscale)"),
    "results/ablation_v3": ("ablation-v3", "AIHUB-H1", "Ablation V3"),
    "results/ablation_v4": ("ablation-v4", "AIHUB-H1", "Ablation V4 (CATFT-CrossAttn)"),
    "results/catft": ("catft-v5", "AIHUB-H3", "CATFT V5 cross-attention 비교군"),
    "results/v2plus": ("v2plus-seed42", "AIHUB-H1", "V2+ 본 모델 (논문 주 결과, seed 라벨 42)"),
    "results/external_baselines/TimesNet": (
        "timesnet-simplified", "AIHUB-H3",
        "TimesNet 축약 구현 비교. 주기식 결함 있음 (연구노트 #16 §3.5)"),
    "results/external_baselines/PatchTST": (
        "patchtst", "AIHUB-H3", "PatchTST 비교"),
    "results/paper_experiments/v2a_multiscale_only": (
        "v2a-multiscale-only", "AIHUB-H1", "단독 요소: multi-scale 차분만"),
    "results/paper_experiments/v2b_se_only": (
        "v2b-se-only", "AIHUB-H1", "단독 요소: SE channel attention만"),
    "results/paper_experiments/v2c_supcon_only": (
        "v2c-supcon-only", "AIHUB-H2", "단독 요소: SupCon만"),
    "results/paper_experiments/v2plus_seed123": (
        "v2plus-seed123", "AIHUB-H1", "V2+ 반복 실행 (seed 라벨 123)"),
    "results/paper_experiments/v2plus_seed456": (
        "v2plus-seed456", "AIHUB-H1", "V2+ 반복 실행 (seed 라벨 456)"),
    "results/paper_experiments/v2plus_sensor_only": (
        "v2plus-sensor-only", "AIHUB-H1", "열화상 제외 sensor-only"),
    "results/paper_experiments/lag_1_3_7": (
        "lag-1-3-7", "AIHUB-H1", "lag 조합 1/3/7"),
    "results/paper_experiments/lag_1_10_20": (
        "lag-1-10-20", "AIHUB-H1", "lag 조합 1/10/20"),
}


def nm_errors(cm) -> str:
    """Normal<->Mild confusions from the stored confusion matrix."""
    if not cm:
        return NO_RECORD
    try:
        return str(int(cm[0][1]) + int(cm[1][0]))
    except (IndexError, TypeError, ValueError):
        return NO_RECORD


def legacy_rows() -> list[dict]:
    rows = []
    for rel, (slug, hyp, purpose) in sorted(LEGACY.items()):
        path = REPO_ROOT / rel / "results.json"
        if not path.is_file():
            continue
        d = json.loads(path.read_text())
        args = d.get("args") or d.get("hparams") or {}
        params = d.get("model_params") or d.get("params")
        seed = d.get("seed")

        secondary = [
            f"val_accuracy={d.get('val_accuracy')}" if d.get("val_accuracy") is not None else "",
            f"nm_errors={nm_errors(d.get('confusion_matrix'))}",
            f"best_epoch={d.get('best_epoch', NO_RECORD)}",
            f"params={params if params is not None else NO_RECORD}",
            f"epochs={args.get('epochs', NO_RECORD)}",
        ]
        rows.append({
            "experiment_id": f"EXP-LEGACY-{slug}",
            "hypothesis_id": hyp,
            "status": "completed",
            "purpose": purpose,
            # 원본 AI Hub subset. 버전 태그가 기록된 적이 없다.
            "data_version": "aihub_71802_released_subset (버전 태그 no_record)",
            "raw_manifest_sha256": NO_RECORD,
            "annotation_version": "provider labels (annotations[0].tagging[0].state)",
            "split_manifest_sha256": NO_RECORD,
            "quality_policy_version": NA + ": legacy 파이프라인에 품질 정책 없음",
            "input_schema_version": "SENSOR_CHANNELS 8ch + thermal 120x160 (버전 태그 없음)",
            "code_commit": NO_RECORD,
            "git_dirty": NO_RECORD,
            "environment_profile": NO_RECORD,
            "environment_record": NO_RECORD,
            "config_path": NO_RECORD,
            "seeds": str(seed) if seed is not None else NO_RECORD,
            # F03: 전달된 seed 는 초기 가중치를 통제하지 못했다 (연구노트 #16 §10 LV-2)
            "seed_controls_init": "false (F03)",
            "primary_metric": "val_macro_f1",
            "primary_metric_value": d.get("val_f1_macro", NO_RECORD),
            "secondary_metrics": "; ".join(x for x in secondary if x),
            "comparison": NO_RECORD,
            "stopping_rule": "best val_macro_f1 checkpoint (early stopping 없음)",
            "result_location": rel,
            "limitations": ("val 4대(agv17,agv18,oht17,oht18) 를 모델 선택과 최종 보고에 재사용; "
                            "git_dirty/environment 미기록으로 clean revision 증명 불가"),
            "decision": "연구노트 #16 및 claim_evidence.csv 참조",
        })
    return rows


def run_rows_for_plan(e: dict) -> list[dict]:
    """One row per completed run listed in a plan entry's `result_dirs`.

    §11: a plan may map to several seed runs, and run ids must not be confused with the
    hypothesis id. Numbers come from each run's results.json (never typed by hand); a
    directory that does not exist yet yields no row, so a plan stays visibly incomplete.
    """
    rows = []
    for rel in e.get("result_dirs") or []:
        path = REPO_ROOT / rel / "results.json"
        if not path.is_file():
            continue
        d = json.loads(path.read_text())
        prov = d.get("provenance") or {}
        seed = d.get("seed", d.get("args", {}).get("seed"))
        # A factorial plan has several conditions per seed; the condition recorded by the
        # run itself goes into the run id so rows stay distinct (EXP-.../ms1_se0_sc1/seed42).
        cond = d.get("condition")
        run_id = f"{e['experiment_id']}/{cond}/seed{seed}" if cond else f"{e['experiment_id']}/seed{seed}"
        rows.append({
            "experiment_id": run_id,
            "hypothesis_id": e.get("hypothesis_id") or NO_RECORD,
            "status": "completed",
            "purpose": e.get("purpose") or "",
            "data_version": e.get("data_version") or "",
            "raw_manifest_sha256": e.get("raw_manifest_sha256") or "",
            "annotation_version": e.get("annotation_version") or "",
            "split_manifest_sha256": e.get("split_manifest_sha256") or "",
            "quality_policy_version": e.get("quality_policy_version") or "",
            "input_schema_version": e.get("input_schema_version") or "",
            "code_commit": prov.get("git_commit") or NO_RECORD,
            "git_dirty": str(prov["git_dirty"]) if prov.get("git_dirty") is not None else NO_RECORD,
            "environment_profile": prov.get("environment_profile") or NO_RECORD,
            "environment_record": (f"{prov.get('hostname')} py{prov.get('python_version')} "
                                   f"torch{prov.get('torch_version')} cuda{prov.get('torch_cuda')} "
                                   f"{prov.get('gpu_name')}") if prov else NO_RECORD,
            "config_path": prov.get("command") or NO_RECORD,
            "seeds": str(seed),
            "seed_controls_init": str(d.get("seed_controls_init", NO_RECORD)),
            "primary_metric": e.get("primary_metric") or "val_macro_f1",
            "primary_metric_value": d.get("val_f1_macro", NO_RECORD),
            "secondary_metrics": "; ".join(x for x in [
                f"val_accuracy={d.get('val_accuracy')}" if d.get("val_accuracy") is not None else "",
                f"nm_errors={nm_errors(d.get('confusion_matrix'))}",
                f"best_epoch={d.get('best_epoch', NO_RECORD)}",
                f"init_sha256={str(d.get('initial_state_sha256', NO_RECORD))[:16]}",
                f"started={prov.get('started_utc','')}", f"finished={prov.get('finished_utc','')}",
            ] if x),
            "comparison": e.get("comparison") or "",
            "stopping_rule": e.get("stopping_rule") or "",
            "result_location": rel,
            "limitations": e.get("limitations") or "",
            "decision": e.get("decision") or "",
        })
    return rows


def plan_rows() -> list[dict]:
    try:
        import yaml
    except ImportError:
        sys.exit("ERROR: pyyaml 이 필요합니다 (계획 manifest 파싱). 설치는 환경 정책을 따르세요.")
    plan = yaml.safe_load(PLAN_YAML.read_text())
    rows = []
    for e in plan.get("experiments") or []:
        run_rows = run_rows_for_plan(e)
        rows.extend(run_rows)
        if run_rows and len(run_rows) >= len(e.get("seeds") or []) and e.get("status") == "completed":
            continue  # 모든 seed 의 실행 행이 있으면 계획 행은 생략한다
        seeds = e.get("seeds")
        rows.append({
            "experiment_id": e["experiment_id"],
            "hypothesis_id": e.get("hypothesis_id") or NO_RECORD,
            "status": e.get("status") or "planned",
            "purpose": e.get("purpose") or "",
            # A plan entry that has since completed may carry provenance of its own
            # (e.g. the G1-S reproduction, which has no results.json). Pass those
            # through verbatim; everything unstated stays empty, never guessed.
            "data_version": e.get("data_version") or "",
            "raw_manifest_sha256": e.get("raw_manifest_sha256") or "",
            "annotation_version": e.get("annotation_version") or "",
            "split_manifest_sha256": e.get("split_manifest_sha256") or "",
            "quality_policy_version": e.get("quality_policy_version") or "",
            "input_schema_version": e.get("input_schema_version") or "",
            "code_commit": e.get("code_commit") or "",
            "git_dirty": e.get("git_dirty") or "",
            "environment_profile": e.get("environment_profile") or "",
            "environment_record": e.get("environment_record") or "",
            "config_path": e.get("config_path") or "",
            "seeds": (", ".join(map(str, seeds)) if isinstance(seeds, list) else (seeds or "")),
            "seed_controls_init": "true (수정안 적용 후)",
            "primary_metric": e.get("primary_metric") or "",
            "primary_metric_value": "",          # 결과 없음 - 비워 둔다
            "secondary_metrics": "",
            "comparison": e.get("comparison") or "",
            "stopping_rule": e.get("stopping_rule") or "",
            "result_location": e.get("result_location") or "",
            "limitations": "; ".join(filter(None, [
                e.get("limitations"),
                ("blocked_by=" + ",".join(e["blocked_by"])) if e.get("blocked_by") else "",
            ])),
            "decision": e.get("decision") or "",
        })
    return rows


def build() -> list[dict]:
    return legacy_rows() + plan_rows()


def render(rows: list[dict]) -> str:
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="파일을 쓰지 않고 최신 상태인지만 확인 (불일치 시 exit 1)")
    args = ap.parse_args()

    rows = build()
    text = render(rows)

    if args.check:
        if not OUT_CSV.is_file():
            print(f"STALE: {OUT_CSV} 없음"); return 1
        if OUT_CSV.read_text() != text:
            print(f"STALE: {OUT_CSV} 가 원본과 다릅니다. 재생성하세요."); return 1
        print(f"OK: {OUT_CSV.name} 최신 ({len(rows)} 행)"); return 0

    OUT_CSV.write_text(text)
    done = sum(1 for r in rows if r["status"] == "completed")
    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)}  총 {len(rows)} 행 "
          f"(completed {done}, 계획/blocked {len(rows) - done})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
