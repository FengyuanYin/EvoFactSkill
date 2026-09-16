from __future__ import annotations

import json
import string

ALLOWED_FIELDS = {"sample", "upstream", "references", "scripts", "summary"}


class TemplateRenderer:
    def __init__(self, *, max_chars: int = 30000):
        self.max_chars = max_chars

    def render(self, template: str, **values) -> str:
        fields = []
        for _, field_name, format_spec, conversion in string.Formatter().parse(template):
            if field_name is None:
                continue
            if field_name not in ALLOWED_FIELDS or "." in field_name or "[" in field_name:
                raise ValueError(f"template field is not allowed: {field_name}")
            if format_spec or conversion:
                raise ValueError("template formatting and conversion are not allowed")
            fields.append(field_name)
        missing = set(fields) - set(values)
        if missing:
            raise ValueError(f"missing template values: {sorted(missing)}")
        serializable = {
            key: value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, default=str)
            for key, value in values.items()
        }
        rendered = template.format_map(serializable)
        if len(rendered) > self.max_chars:
            raise ValueError("rendered template exceeds character limit")
        return rendered
