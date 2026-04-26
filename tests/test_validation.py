from __future__ import annotations

import unittest

from justbuild.prompts import architecture_review_user_prompt, architecture_user_prompt, implementation_user_prompt, specification_user_prompt
from justbuild.validation import JSONValidationError, parse_architecture_plan, parse_fix_plan, parse_implementation_file, parse_implementation_plan, parse_product_specification
from tests.support import debugging_response, default_responses


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

    def test_parse_implementation_plan_allows_missing_notes(self) -> None:
        plan = parse_implementation_plan(
            '{"prototype_kind":"static_web","entrypoint":"index.html","files":[{"path":"index.html","purpose":"entry","required":true}]}'
        )
        self.assertEqual(plan.notes, [])

    def test_parse_implementation_file_allows_missing_notes(self) -> None:
        path, content, notes = parse_implementation_file('{"path":"styles.css","content":":root { --x: 1; }"}')
        self.assertEqual(path, "styles.css")
        self.assertEqual(content, ":root { --x: 1; }")
        self.assertEqual(notes, [])

    def test_parse_implementation_file_rejects_empty_content(self) -> None:
        with self.assertRaises(JSONValidationError):
            parse_implementation_file('{"path":"styles.css","content":""}')

    def test_prompts_emphasize_smallest_recognizable_prototype(self) -> None:
        responses = default_responses()
        spec = parse_product_specification(responses["specification"])
        architecture = parse_architecture_plan(responses["architecture_plan"])

        specification_prompt = specification_user_prompt("Simple task tracker", None)
        architecture_prompt = architecture_user_prompt(spec)
        architecture_review_prompt = architecture_review_user_prompt(spec, architecture)
        implementation_prompt = implementation_user_prompt(spec, architecture, None)

        self.assertIn("smallest recognizable prototype", specification_prompt)
        self.assertIn("smallest recognizable prototype", architecture_review_prompt)
        self.assertIn("smallest recognizable prototype", architecture_prompt)
        self.assertIn("minimum file set", implementation_prompt)


if __name__ == "__main__":
    unittest.main()
