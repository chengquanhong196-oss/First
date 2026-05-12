"""Unit tests for configuration deep merge."""

from mini_swe_agent.config.merger import deep_merge


class TestDeepMerge:
    def test_scalar_override(self):
        base = {"a": 1, "b": 2}
        override = {"a": 10}
        result = deep_merge(base, override)
        assert result == {"a": 10, "b": 2}

    def test_nested_dict_merge(self):
        base = {"model": {"name": "claude", "temp": 0.0}}
        override = {"model": {"temp": 0.7}}
        result = deep_merge(base, override)
        assert result == {"model": {"name": "claude", "temp": 0.7}}

    def test_list_replaced_not_merged(self):
        base = {"tags": [1, 2, 3]}
        override = {"tags": [4, 5]}
        result = deep_merge(base, override)
        assert result == {"tags": [4, 5]}

    def test_none_removes_key(self):
        base = {"a": 1, "b": 2}
        override = {"a": None}
        result = deep_merge(base, override)
        assert "a" not in result
        assert result["b"] == 2

    def test_new_key_added(self):
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_deep_nested_merge(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 10}}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": {"c": 10, "d": 2}}}

    def test_empty_base(self):
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self):
        result = deep_merge({"a": 1}, {})
        assert result == {"a": 1}
