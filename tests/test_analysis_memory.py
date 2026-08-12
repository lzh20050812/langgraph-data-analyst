from config.prompts.report_prompt import _format_memory_for_prompt


def test_memory_prompt_marks_historical_content():
    text = _format_memory_for_prompt([{"query": "流失分析", "score": .8, "document": "策略框架"}])
    assert "相似问题" in text and "策略框架" in text


def test_empty_memory_prompt_is_explicit():
    assert "无可复用" in _format_memory_for_prompt([])
