import math

from ecom.reporting import README_END, README_START, _num, _pct, _pval, _signed_pct, render_reports


def test_filters():
    assert _num(1234567.891, 1) == "1,234,567.9"
    assert _pct(0.1234) == "12.3%"
    assert _signed_pct(0.05) == "+5.0%"
    assert _pval(1e-20) == "< 1e-16"
    assert _pval(0.0004) == "0.0004"
    assert _pval(0.0312) == "0.031"
    assert _num(math.nan) == "n/a" and _pct(None) == "n/a"


def test_readme_block_is_replaced_and_rest_kept(cfg):
    templates = cfg.paths.reports / "templates"
    templates.mkdir(parents=True)
    for name in ("report.md.j2", "business_memo.md.j2", "slides.md.j2"):
        (templates / name).write_text("{{ source }} {{ sample_rate | pct(0) }}\n")
    (templates / "readme_results.md.j2").write_text("rows: {{ funnel | length }}")
    readme = cfg.root / "README.md"
    readme.write_text(f"# Title\n\n{README_START}\nold\n{README_END}\n\nfooter\n")
    written = render_reports(cfg)
    assert readme in written
    text = readme.read_text()
    assert "old" not in text and "rows: 0" in text
    assert text.startswith("# Title") and text.rstrip().endswith("footer")
    assert (cfg.paths.reports / "report.md").read_text().strip() == "synthetic 100%"
