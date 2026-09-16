from __future__ import annotations

import difflib

from evofact.core.package_models import (
    FileOperationKind,
    PackageDiff,
    PackageFileDiff,
    SkillPackage,
)


def diff_packages(before: SkillPackage, after: SkillPackage) -> PackageDiff:
    manifest_changes = {}
    left_manifest = before.manifest.model_dump()
    right_manifest = after.manifest.model_dump()
    for key in sorted(set(left_manifest) | set(right_manifest)):
        if left_manifest.get(key) != right_manifest.get(key):
            manifest_changes[key] = (left_manifest.get(key), right_manifest.get(key))
    left = {item.path: item for item in before.files}
    right = {item.path: item for item in after.files}
    changes = []
    for path in sorted(set(left) | set(right)):
        old, new = left.get(path), right.get(path)
        if old is None:
            changes.append(
                PackageFileDiff(
                    FileOperationKind.ADD, path, new_digest=new.digest, new_size=len(new.content)
                )
            )
        elif new is None:
            changes.append(
                PackageFileDiff(
                    FileOperationKind.DELETE, path, old_digest=old.digest, old_size=len(old.content)
                )
            )
        elif (
            old.digest != new.digest
            or old.media_type != new.media_type
            or old.executable != new.executable
        ):
            text_diff = None
            if old.media_type.startswith("text/") and new.media_type.startswith("text/"):
                text_diff = "".join(
                    difflib.unified_diff(
                        old.content.decode("utf-8").splitlines(True),
                        new.content.decode("utf-8").splitlines(True),
                        fromfile=path,
                        tofile=path,
                    )
                )
            changes.append(
                PackageFileDiff(
                    FileOperationKind.UPDATE,
                    path,
                    old_digest=old.digest,
                    new_digest=new.digest,
                    old_size=len(old.content),
                    new_size=len(new.content),
                    text_diff=text_diff,
                )
            )
    return PackageDiff(
        before.package_digest, after.package_digest, manifest_changes, tuple(changes)
    )
