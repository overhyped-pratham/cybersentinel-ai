import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.replay_service import ReplayService
from ml.pipeline.detection_pipeline import DetectionPipeline
from ml.state.state_builder import FEATURE_NAMES

sys.stdout.reconfigure(encoding="utf-8")

print("Loading ReplayService data (384 windows)...")
svc = ReplayService()
svc._load_data()
df = svc._states_df
print(f"Loaded {len(df)} windows from {df['scenario_id'].nunique()} scenarios.")

pipeline = DetectionPipeline()

matches = []
for idx, row in df.iterrows():
    raw_vec = [float(row[col]) for col in FEATURE_NAMES]
    res = pipeline.process_state(raw_vec, is_scaled=False)
    clf_cat = res.current_stage
    clf_conf = res.known_classifier["confidence"]
    anom = res.novelty_detector["anomaly_score"]
    is_nov = res.is_novel
    verdict = res.threat_classification
    
    # Model Disagreement: Classifier uncertain/benign (< 0.70) while Anomaly is HIGH (>= 0.70)
    # OR Classifier says BENIGN while Anomaly is elevated (is_nov = True)
    if (clf_cat == "BENIGN" or clf_conf < 0.70) and is_nov and anom >= 0.70:
        matches.append({
            "idx": idx,
            "scenario": row.get("scenario_id", "unknown"),
            "true_stage": row.get("stage", "unknown"),
            "clf_cat": clf_cat,
            "clf_conf": clf_conf,
            "anom": anom,
            "risk": res.risk_score,
            "verdict": verdict,
            "features": raw_vec,
        })
        print(f"Match win {idx} ({row.get('scenario_id')}): True={row.get('stage')}, Clf={clf_cat} (conf={clf_conf:.2f}), Anomaly={anom:.2f}, Verdict={verdict}, Risk={res.risk_score:.1f}")

print(f"\nTotal real disagreement samples found: {len(matches)}")
if matches:
    print(f"\nTop Match:")
    top = matches[0]
    print(f"Scenario: {top['scenario']}, True Stage: {top['true_stage']}")
    print(f"Classifier: {top['clf_cat']} (Conf: {top['clf_conf']:.3f})")
    print(f"Anomaly: {top['anom']:.3f}")
    print(f"Risk: {top['risk']:.1f}")
    print(f"Verdict: {top['verdict']}")
    print(f"Raw Vector: {top['features']}")
