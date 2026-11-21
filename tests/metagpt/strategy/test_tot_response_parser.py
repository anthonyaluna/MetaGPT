"""Offline controls for model responses entering the Tree-of-Thought parser."""

import asyncio
from types import SimpleNamespace

import pytest

from metagpt.strategy.base import ThoughtNode, ThoughtTree
from metagpt.strategy.tot import ThoughtSolverBase


class StubParser:
    def propose(self, current_state, **_kwargs):
        return current_state


class StubLLM:
    def __init__(self, response):
        self.response = response

    async def aask(self, msg):
        assert "strictly a list of nodes" in msg
        return self.response


def generate(response):
    root = ThoughtNode("root")
    tree = ThoughtTree(root)
    solver = SimpleNamespace(
        config=SimpleNamespace(parser=StubParser(), n_generate_sample=5),
        llm=StubLLM(response),
        thought_tree=tree,
    )
    return asyncio.run(ThoughtSolverBase.generate_thoughts(solver, "safe state", root)), root


@pytest.mark.parametrize(
    "response",
    [
        '[{"node_id":"1","node_state_instruction":"first"}]',
        '```json\n[{"node_id":1,"node_state_instruction":"first"}]\n```',
        "[{'node_id': '1', 'node_state_instruction': 'first'}]",
    ],
)
def test_thought_response_preserves_json_and_legacy_literal_lists(response):
    nodes, root = generate(response)

    assert [(node.id, node.name) for node in nodes] == [(1, "first")]
    assert nodes[0].parent is root


@pytest.mark.parametrize("fenced", [False, True])
def test_thought_response_never_evaluates_expressions(tmp_path, fenced):
    marker = tmp_path / "eval-marker"
    response = (
        f"(__import__('pathlib').Path({str(marker)!r}).write_text('EXECUTED'), "
        "[{'node_id': '1', 'node_state_instruction': 'first'}])[1]"
    )
    if fenced:
        response = "```python\n" + response + "\n```"

    with pytest.raises(ValueError):
        generate(response)

    assert not marker.exists()


def test_thought_response_rejects_invalid_shape_before_creating_nodes():
    response = '[{"node_id":"1","node_state_instruction":"first"}, {"node_id":"2"}]'
    root = ThoughtNode("root")
    tree = ThoughtTree(root)
    solver = SimpleNamespace(
        config=SimpleNamespace(parser=StubParser(), n_generate_sample=5),
        llm=StubLLM(response),
        thought_tree=tree,
    )

    with pytest.raises(ValueError):
        asyncio.run(ThoughtSolverBase.generate_thoughts(solver, "safe state", root))

    assert root.children == ()


@pytest.mark.parametrize(
    "response",
    [
        '{"node_id":"1","node_state_instruction":"first"}',
        '[{"node_id":true,"node_state_instruction":"first"}]',
        '[{"node_id":"1","node_state_instruction":123}]',
        '[{"node_id":"not-a-number","node_state_instruction":"first"}]',
    ],
)
def test_thought_response_rejects_wrong_data_types(response):
    with pytest.raises(ValueError):
        generate(response)
