# Open-RL

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/Open-RL)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/TuringEnterprises/Open-RL)

## Description

Open-RL is an environment sourced from the [Turing Enterprises](https://huggingface.co/TuringEnterprises) dataset for evaluating AI agents on self-contained, verifiable STEM reasoning problems. Problems span Physics, Mathematics, Chemistry, and Biology, requiring multi-step reasoning, symbolic manipulation, and numerical computation.

## Capabilities

- Solving complex STEM problems requiring multi-step reasoning
- Symbolic manipulation and algebraic simplification
- Numerical computation and derivations
- Cross-domain scientific reasoning

## License

[MIT](https://opensource.org/licenses/MIT)

## Tasks

There are 40 tasks in the train split, covering:

- **Physics**: Astrophysics, electromagnetism, quantum mechanics, condensed matter, classical mechanics
- **Mathematics**: Number theory, special functions, combinatorics, analysis
- **Chemistry**: General, medicinal, inorganic chemistry
- **Biology**: Molecular biology, immunology, neurobiology, physiology, microbiology

## Reward Structure

Binary reward (0 or 1) based on answer correctness. An LLM grader (gpt-5-mini) checks semantic/symbolic equivalence between the submitted answer and ground truth.

## Data

Data is sourced from the [TuringEnterprises/Open-RL](https://huggingface.co/datasets/TuringEnterprises/Open-RL) dataset on Hugging Face.

## Tools

Single tool:

- `answer(answer: str)` - Submit your solution to be graded

## Time Horizon

Open-RL is a single-turn environment. Each task requires exactly one tool call to submit an answer. The agent receives a problem, performs reasoning, and submits its final answer.

## Other Environment Requirements

- **OpenAI API Key**: Required for LLM-based grading

## Safety

Open-RL presents minimal safety risks. Agents interact only with static STEM problems and submit text answers for grading. There is no network access, filesystem interaction, or execution of agent-generated code. The environment does not involve real-world actions, external systems, or other agents.

## Citations

```bibtex
@dataset{OpenRL2024,
  author    = {Turing Enterprises},
  title     = {Open-RL: Self-contained, verifiable STEM reasoning problems},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/TuringEnterprises/Open-RL}
}
```
