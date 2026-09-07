import hashlib
from collections import defaultdict
from evofact.core.models import AttributionReport

def cluster_reports(reports:list[AttributionReport])->dict[str,list[AttributionReport]]:
    groups=defaultdict(list)
    for report in reports:
        key="+".join(sorted(x.value for x in report.error_types)) or "success"
        groups[f"cluster-{hashlib.sha1(key.encode()).hexdigest()[:8]}"] .append(report)
    return dict(groups)
