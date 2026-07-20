"""Agent loop for Open-RL.

Open-RL uses a one-argument @terminal tool, and it is the environment's only
tool — so the model is given NO tools at all. It reads the STEM problem and
replies with its answer as an ordinary message; the harness routes that message
text to session.call_terminal_tool(), which grades it for mathematical /
scientific equivalence against the ground truth with an LLM judge.

Because list_tools() is empty, the `tools` argument is omitted entirely rather
than passed as [] (an empty tools array is rejected by some providers).

Runs against the deployed environment by default; set LOCAL=1 to point at a
local `python server.py` on port 8080.

Writes a trajectory to open_rl_trajectory.jsonl.
"""

import asyncio
import json
import os
from datetime import datetime, timezone

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

TRAJECTORY_PATH = "open_rl_trajectory.jsonl"


def _text_of(response) -> str:
    parts = []
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    parts.append(block.text)
    return "\n".join(parts).strip()


async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
    ENV_NAME = "GeneralReasoning/Open-RL"
    SPLIT = os.environ.get("SPLIT", "train")
    NUM_TASKS = int(os.environ.get("NUM_TASKS", "2"))
    MAX_TURNS = int(os.environ.get("MAX_TURNS", "10"))
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    base_url = "http://localhost:8080" if os.environ.get("LOCAL") else None
    environment = or_client.environments.get(name=ENV_NAME, base_url=base_url)
    print(f"Environment: {ENV_NAME} ({base_url or 'deployed'})")

    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")
    terminal_tool = await environment.terminal_tool()

    print(f"Found {len(tasks)} tasks in split {SPLIT!r}")
    print(f"Tools visible to the model: {[t['name'] for t in tools]}")
    print(f"Terminal tool (hidden): {terminal_tool}")

    traj = open(TRAJECTORY_PATH, "w")

    def record(kind: str, **fields):
        traj.write(json.dumps({
            "kind": kind,
            "ts": datetime.now(timezone.utc).isoformat(),
            **fields,
        }) + "\n")
        traj.flush()

    record(
        "config",
        model=MODEL_NAME,
        env=ENV_NAME,
        base_url=base_url or "deployed",
        split=SPLIT,
        visible_tools=[t["name"] for t in tools],
        terminal_tool=None if terminal_tool is None else {
            "name": terminal_tool.name,
            "arg": terminal_tool.arg,
            "description": terminal_tool.description,
        },
    )

    rewards = []

    for task in tasks[:NUM_TASKS]:
        encounter = task.task_spec.get("id", "?")
        print(f"\n=== Task {encounter} ===")

        async with environment.session(
            task=task,
            secrets={"openai_api_key": OPENAI_API_KEY},
        ) as session:
            assistant_ends_rollout = await session.is_assistant_message_final()
            session_tools = await session.list_tools()
            print(f"is_assistant_message_final() -> {assistant_ends_rollout}")
            print(f"session.list_tools() -> {[t.name for t in session_tools]}")
            assert "answer" not in [t.name for t in session_tools], \
                "terminal tool leaked into the model's tool list"

            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]

            record(
                "task_start",
                task_id=encounter,
                is_assistant_message_final=assistant_ends_rollout,
                session_tools=[t.name for t in session_tools],
                prompt=prompt[0].text,
            )

            reward = None
            turn = 0

            while turn < MAX_TURNS:
                turn += 1

                # Omit `tools` when the environment exposes none.
                kwargs = {"model": MODEL_NAME, "input": input_list}
                if tools:
                    kwargs["tools"] = tools
                response = await oai_client.responses.create(**kwargs)
                input_list += response.output

                calls = [i for i in response.output if i.type == "function_call"]
                if calls:
                    for item in calls:
                        args = json.loads(str(item.arguments))
                        tool_result = await session.call_tool(item.name, args)
                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text,
                        })
                        record("tool_call", task_id=encounter, turn=turn,
                               tool=item.name, arguments=args,
                               output=tool_result.blocks[0].text)
                    continue

                answer_text = _text_of(response)
                print(f"[{turn}] answer: {len(answer_text)} chars — {answer_text[:160]!r}")
                record("assistant_final_message", task_id=encounter,
                       turn=turn, text=answer_text)

                if not assistant_ends_rollout:
                    print("Not a terminal-tool environment; stopping.")
                    break

                out = await session.call_terminal_tool(answer_text)
                reward = out.reward
                print(f"call_terminal_tool() -> reward={reward} finished={out.finished}")
                print(out.blocks[0].text[:600])
                record("terminal_tool_result", task_id=encounter, turn=turn,
                       submitted=answer_text, reward=out.reward, finished=out.finished,
                       output=out.blocks[0].text, metadata=out.metadata)
                break

            rewards.append(reward)
            record("task_end", task_id=encounter, turns=turn, reward=reward)

    scored = [r for r in rewards if r is not None]
    summary = {
        "num_tasks": len(rewards),
        "num_scored": len(scored),
        "mean_reward": (sum(scored) / len(scored)) if scored else None,
        "rewards": rewards,
    }
    record("summary", **summary)
    traj.close()

    print(f"\n=== Summary ===\n{json.dumps(summary, indent=2)}")
    print(f"Trajectory written to {TRAJECTORY_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
