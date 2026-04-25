from paddle_pyfi.domains import available_domains, load_domain


def test_available_domains_include_finance() -> None:
    assert "finance" in available_domains()


def test_load_finance_domain() -> None:
    domain = load_domain("finance")
    assert domain.name == "finance"
    assert "extract_metrics" in domain.tasks
    assert "research_conclusion" in domain.output_schema
