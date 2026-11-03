from __future__ import annotations

import copy

import pytest

from showlib.schemas import PRODUCTS, load_schema, render_markdown


def _row(markdown: str, name: str) -> str:
    """The table row for a field named ``name``."""
    return next(
        line for line in markdown.splitlines() if line.startswith(f"| `{name}` ")
    )


@pytest.mark.parametrize("product", PRODUCTS)
def test_title_and_version(product):
    schema = load_schema(product)
    markdown = render_markdown(schema)
    assert markdown.startswith(f"# {schema['product']}\n")
    assert f"Schema version {schema['schema_version']}" in markdown


def test_variable_row():
    row = _row(render_markdown(load_schema("l1a")), "image")
    assert "time, los, pixelcolumn" in row
    assert "`count`" in row
    assert "| yes |" in row
    assert "Raw detector signal" in row


def test_scalar_range_and_default():
    markdown = render_markdown(load_schema("l1a"))
    assert "scalar" not in _row(markdown, "image")
    assert "[-90, 90]" in _row(markdown, "tangent_latitude")
    assert "no (default 1)" in _row(markdown, "num_averaged_exposures")
    assert "≥ 1" in _row(markdown, "num_averaged_exposures")


def test_conditional_requirement_and_allowed_values():
    markdown = render_markdown(load_schema("l1b"))
    assert "when apodization_window = kaiser" in _row(markdown, "apodization_beta")
    assert "rectangular, kaiser" in _row(markdown, "apodization_window")


def test_groups_are_rendered():
    markdown = render_markdown(load_schema("l1a"))
    assert "## Group `simulation`" in markdown
    assert "time, los, pixelcolumn" in _row(markdown, "image_noiseless")


def test_model_is_rendered_as_text_blocks():
    markdown = render_markdown(load_schema("cal_db"))
    assert "## Model" in markdown
    assert "```text\n" in markdown
    assert "C(y,x,\u03bd)" in markdown


def test_constraints_are_listed():
    markdown = render_markdown(load_schema("cal_db"))
    assert "- **`samples-one-sided`**:" in markdown


def test_open_questions_are_listed_only_when_present():
    schema = copy.deepcopy(load_schema("l1b"))
    assert "## Open questions" not in render_markdown(schema)

    schema["open_questions"] = [
        {"id": "Q-1", "question": "Why?", "context": "Because."}
    ]
    markdown = render_markdown(schema)
    assert "## Open questions" in markdown
    assert "**Q-1**: Why? Because." in markdown


def test_pipes_in_text_do_not_break_tables():
    schema = copy.deepcopy(load_schema("l1b"))
    schema["variables"]["radiance"]["description"] = "uses |x| here"
    assert "uses \\|x\\| here" in _row(render_markdown(schema), "radiance")
