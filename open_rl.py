import json
import re
from pathlib import Path
from typing import List

import openai
from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, ToolOutput, tool, TextBlock
from prompts import GRADER_TEMPLATE


def parse_grader_response(response: str) -> dict:
    """Extract JSON from <answer> tags in grader response."""
    # Try to extract content from <answer> tags
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if answer_match:
        json_str = answer_match.group(1).strip()
    else:
        # Fall back to entire response if no tags found
        json_str = response.strip()

    # Clean up any markdown code blocks
    json_str = re.sub(r"^```json\s*|\s*```$", "", json_str.strip())

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def load_data() -> tuple[list[dict], dict[str, str]]:
    """Load dataset from local file or production path."""
    # Production path (OpenReward platform)
    prod_path = Path("/orwd_data/data.json")
    # Local development path (same directory as this file)
    local_path = Path(__file__).parent / "data.json"

    if prod_path.exists():
        data_path = prod_path
    elif local_path.exists():
        data_path = local_path
    else:
        raise FileNotFoundError(
            f"Data file not found. Checked:\n"
            f"  - {prod_path}\n"
            f"  - {local_path}\n"
            "Run 'python download_data.py' to download the dataset."
        )

    with open(data_path, "r") as f:
        data = json.load(f)

    tasks = []
    ground_truth = {}

    for row in data:
        task_id = row["conversation_id"]
        tasks.append({
            "id": task_id,
            "domain": row["domain"],
            "sub_domain": row["sub_domain"],
            "question": row["question"],
        })
        ground_truth[task_id] = row["answer"]

    return tasks, ground_truth


# Load data at module level
tasks, ground_truth = load_data()


class TaskSpec(BaseModel):
    id: str
    domain: str
    sub_domain: str
    question: str


class AnswerInput(BaseModel, extra="forbid"):
    """Submit your answer to the STEM problem."""
    answer: str


class OpenRL(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)

        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via secrets parameter")

        self.client = openai.AsyncClient(api_key=api_key)
        self.ground_truth = ground_truth[self.validated.id]

    async def get_prompt(self) -> List[TextBlock]:
        prompt = f"""{self.validated.question}

When you have your answer, submit it using the answer tool."""
        return [TextBlock(text=prompt)]

    async def _grade_answer(self, submitted_answer: str) -> dict:
        grader_prompt = GRADER_TEMPLATE.replace(
            "<<submitted_answer>>", submitted_answer
        ).replace(
            "<<ground_truth>>", self.ground_truth
        )

        while True:
            res = await self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": grader_prompt}],
                stream=False
            )
            grading_response = res.choices[0].message.content or ""
            grading_dict = parse_grader_response(grading_response)

            if "is_correct" in grading_dict and isinstance(grading_dict["is_correct"], bool):
                break
            print("Grading failed due to bad JSON output, retrying...")

        return grading_dict

    @tool
    async def answer(self, params: AnswerInput) -> ToolOutput:
        """Submit your answer to be graded."""
        grader_output = await self._grade_answer(params.answer)

        is_correct = grader_output.get("is_correct", False)
        explanation = grader_output.get("explanation", "")
        reward = 1.0 if is_correct else 0.0

        result_text = f"{'Correct!' if is_correct else 'Incorrect.'}\n{explanation}"

        return ToolOutput(
            metadata={
                "submitted_answer": params.answer,
                "is_correct": is_correct,
                "explanation": explanation,
            },
            blocks=[TextBlock(text=result_text)],
            reward=reward,
            finished=True
        )

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return tasks  # type: ignore
        raise ValueError(f"Unknown split: {split}")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]
