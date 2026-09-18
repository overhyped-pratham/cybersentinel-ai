import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("dashboard/index.html", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line_s = line.strip()
        if any(keyword in line_s for keyword in [
            'class="soc-sidebar', 'sidebar-nav-item', 'class="card-header',
            'id="sec-', 'id="panel-', 'id="tab-', 'class="tab-pane',
            'Threat Feed', 'World Model', 'Attack Story', 'Forecasting',
            'Evidence', 'Adaptive'
        ]):
            print(f"L{i}: {line_s[:100]}")
