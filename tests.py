import json
import re
from types import SimpleNamespace

import pytest

from open_rl import AnswerInput, OpenRL, ground_truth, parse_grader_response

TASKS = OpenRL.list_tasks("train")


class FakeCompletions:
    """Stands in for the grader LLM: equivalence is exact string match, and the
    explanation quotes the ground truth the way the real grader does."""

    def __init__(self) -> None:
        self.calls = 0

    async def create(self, model, messages, stream):
        self.calls += 1
        prompt = messages[0]["content"]
        truth = re.search(r"# Ground Truth Answer\n(.*?)\n\n# Submitted Answer", prompt, re.DOTALL).group(1)
        submitted = re.search(r"# Submitted Answer\n(.*?)\n\n# Instructions", prompt, re.DOTALL).group(1)
        is_correct = submitted.strip() == truth.strip()
        verdict = {
            "is_correct": is_correct,
            "explanation": f"The submitted answer {submitted!r} {'matches' if is_correct else 'differs from'} "
                           f"the ground truth {truth!r}.",
        }
        content = f"<reasoning>Compared.</reasoning>\n<answer>\n{json.dumps(verdict)}\n</answer>"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def make_env(task) -> tuple[OpenRL, FakeCompletions]:
    env = OpenRL(task_spec=task, secrets={"openai_api_key": "test-key"})
    completions = FakeCompletions()
    env.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return env, completions


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS[:5], ids=lambda t: t["id"])
async def test_gold(task):
    env, _ = make_env(task)
    result = await env.answer(AnswerInput(answer=ground_truth[task["id"]]))
    assert result.reward == 1.0
    assert result.finished
    assert result.metadata["is_correct"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
async def test_incorrect_feedback_does_not_reveal_ground_truth(task):
    env, _ = make_env(task)
    truth = ground_truth[task["id"]]
    result = await env.answer(AnswerInput(answer="I could not work this out."))
    assert result.reward == 0.0
    assert result.finished
    assert result.metadata["is_correct"] is False
    text = " ".join(b.text for b in result.blocks)
    assert truth not in text
    assert truth not in json.dumps(result.metadata, ensure_ascii=False)
    assert "explanation" not in result.metadata


@pytest.mark.asyncio
async def test_repeat_submission_is_not_regraded():
    task = TASKS[0]
    env, completions = make_env(task)
    await env.answer(AnswerInput(answer=ground_truth[task["id"]]))
    result = await env.answer(AnswerInput(answer=ground_truth[task["id"]]))
    assert result.reward < 0
    assert result.finished
    assert completions.calls == 1


def test_parse_grader_response():
    assert parse_grader_response('<answer>\n{"is_correct": true}\n</answer>') == {"is_correct": True}
    assert parse_grader_response('<answer>```json\n{"is_correct": false}\n```</answer>') == {"is_correct": False}
    assert parse_grader_response("not json") == {}
