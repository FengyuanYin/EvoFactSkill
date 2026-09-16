from __future__ import annotations

from evofact.core.package_models import SkillPackage

from .scanner import SafetyReport, scan_resources


def scan_package(package: SkillPackage) -> SafetyReport:
    scripts: dict[str, str] = {}
    for item in package.files:
        if item.path.startswith("scripts/"):
            try:
                scripts[item.path] = item.content.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                return SafetyReport("blocked", (f"script is not UTF-8: {item.path}",))
    report = scan_resources(scripts)
    declared = package.manifest.entrypoints.script
    if scripts and declared not in scripts:
        return SafetyReport(
            "blocked", (*report.findings, "manifest script entrypoint is missing or undeclared")
        )
    if scripts and report.level == "safe":
        return SafetyReport("review_required", report.findings)
    return report
