from pathlib import Path

from evofact.core.models import SkillSpec, SkillStatus

from .package_adapter import package_to_skill_spec
from .package_loader import load_package


def load_skill_package(directory: Path, *, status: SkillStatus = SkillStatus.ACTIVE) -> SkillSpec:
    """函数作用：读取并转换 `load_skill_package` 所表示的数据，供当前模块后续流程使用。
    输入要求：`directory`（Path）需符合函数签名约定；`status`（SkillStatus，默认 `SkillStatus.ACTIVE`）需以关键字传入并符合签名约定。
    输出：返回 `SkillSpec` 类型结果；校验或下游调用失败时异常向上传递。"""
    return package_to_skill_spec(load_package(directory, status=status))
