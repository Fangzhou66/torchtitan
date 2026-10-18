# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

from harbor.agents.terminus_2.terminus_xml_plain_parser import (
    ParseResult,
    TerminusXMLPlainParser,
)

# Harbor returns no error and no warning when <commands> has text but no
# <keystrokes>, so the model never learns why nothing ran.
SILENT_NOOP_WARNING = (
    "Commands section has text but no <keystrokes> elements, so nothing was "
    'executed. Wrap each command as <keystrokes duration="1.0">your command\n'
    "</keystrokes>."
)


class SafeTerminusXMLParser(TerminusXMLPlainParser):
    """Skip empty tag names in diagnostics while preserving raw command text,
    and warn when a non-empty <commands> section yields no keystrokes."""

    def parse_response(self, response: str) -> ParseResult:
        result = super().parse_response(response)
        if result.commands or result.error or result.is_task_complete:
            return result
        content = self._extract_response_content(response)
        if not self._extract_sections(content, []).get("commands"):
            return result
        # Same channel as harbor's empty-commands warning.
        note = f"- {SILENT_NOOP_WARNING}"
        result.warning = f"{result.warning}\n{note}" if result.warning else note
        return result

    def _find_top_level_tags(self, content: str) -> list[str]:
        # Match the upstream scanner's first '<' through next '>' boundaries.
        diagnostic_content = re.sub(
            r"<[^>]*>",
            lambda match: match[0] if match[0][1:-1].strip() else "",
            content,
        )
        return super()._find_top_level_tags(diagnostic_content)
