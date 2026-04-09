from __future__ import annotations

import unittest

from justbuild.validation import JSONValidationError, parse_fix_plan
from tests.support import debugging_response


class ValidationTests(unittest.TestCase):
    def test_parse_fix_plan_success(self) -> None:
        fix_plan = parse_fix_plan(debugging_response(failure_groups=["schema_mismatch"]))
        self.assertEqual(fix_plan.failure_groups, ["schema_mismatch"])
        self.assertTrue(fix_plan.file_changes)

    def test_parse_fix_plan_rejects_invalid_failure_group(self) -> None:
        with self.assertRaises(JSONValidationError):
            parse_fix_plan(
                '{"file_changes":["Update app.js"],"root_cause":"Bad output","strategy":"Retry","failure_groups":["unknown"],"priority_order":["app.js"]}'
            )


if __name__ == "__main__":
    unittest.main()
