"""Anti-hardcoding audit script for Phase 12."""
import os

patterns = [
    ("if stage ==", "Conditional branch on stage name"),
    ("if scenario", "Conditional branch on scenario ID"),
    ("next_stage = ", "Hardcoded next_stage assignment"),
    ("attack_probability = 0", "Hardcoded attack probability literal"),
    ("risk_score = ", "Hardcoded risk score assignment"),
    ("prediction = {", "Hardcoded prediction dict"),
    ("fake_", "Fake/demo data marker"),
    ("demo_data", "Demo data marker"),
    ("= 0.99", "Hardcoded high probability"),
    ("static_prob", "Static probability variable"),
]

dirs = ["backend", "ml", "agent", "mitre", "network", "frontend"]
findings = {}

for pattern, desc in patterns:
    hits = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".py"):
                    fpath = os.path.join(root, f)
                    try:
                        with open(fpath, encoding="utf-8") as fh:
                            for lno, line in enumerate(fh, 1):
                                if pattern in line:
                                    stripped = line.strip()
                                    # Skip comment lines
                                    if not stripped.startswith("#"):
                                        hits.append(f"{fpath}:{lno}: {stripped[:100]}")
                    except Exception:
                        pass
    findings[pattern] = {"desc": desc, "hits": hits}
    status = "CLEAN" if not hits else f"{len(hits)} HIT(S)"
    print(f"[{status}] {pattern!r} -- {desc}")
    for h in hits:
        print(f"  -> {h}")

print("\nDONE -- Anti-hardcoding audit complete.")
