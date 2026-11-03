# Open-RL

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/Open-RL) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/TuringEnterprises/Open-RL)

## Description

Open-RL is an environment sourced from the [Turing Enterprises](https://huggingface.co/TuringEnterprises) dataset for evaluating AI agents on self-contained, verifiable STEM reasoning problems. Problems span Physics, Mathematics, Chemistry, and Biology, requiring multi-step reasoning, symbolic manipulation, and numerical computation.

## Capabilities

- Solving complex STEM problems requiring multi-step reasoning
- Symbolic manipulation and algebraic simplification
- Numerical computation and derivations
- Cross-domain scientific reasoning

## Compute Requirements

Open-RL does not require a sandbox. Agents have no file system or code execution access.

## License

[MIT](https://opensource.org/licenses/MIT). The underlying TuringEnterprises/Open-RL dataset is also released under MIT.

## Tasks

There are 40 tasks in the `train` split, covering:

- **Physics**: Astrophysics, electromagnetism, quantum mechanics, condensed matter, classical mechanics
- **Mathematics**: Number theory, special functions, combinatorics, analysis
- **Chemistry**: General, medicinal, inorganic chemistry
- **Biology**: Molecular biology, immunology, neurobiology, physiology, microbiology

## Reward Structure

Binary reward (0 or 1) based on answer correctness. An LLM grader (gpt-5-mini) checks semantic/symbolic equivalence between the final message and the ground truth. The grader is re-sampled until it returns a valid verdict. Only the first graded answer is rewarded; a repeat submission gets a reward of -0.1.

## Data

Data is sourced from the [TuringEnterprises/Open-RL](https://huggingface.co/datasets/TuringEnterprises/Open-RL) dataset on Hugging Face and served as a JSON file from the OpenReward platform.

## Tools

None. The model is given no tools: it answers the problem as an ordinary message, and that message ends the rollout.

Grading runs through a hidden terminal tool, `answer`, which sends the message text to an LLM grader (gpt-5-mini) alongside the ground truth.

## Time Horizon

Open-RL is a single-turn environment. The agent receives a problem, performs reasoning, and replies with its final answer as an ordinary message, which ends the rollout.

## Other Environment Requirements

The LLM grader needs an OpenAI API key, passed as the `openai_api_key` secret.

## Safety

Open-RL presents minimal safety risks. Agents interact only with static STEM problems and submit text answers for grading. There is no network access, filesystem interaction, or execution of agent-generated code. The environment does not involve real-world actions, external systems, or other agents.

## Citations

```bibtex
@dataset{turing_2026_open_rl,
  title        = {Open-RL},
  author       = {Saurabh Patil and Anshuman Lall and Marko Pavlovic and Chinmayee Shukla and Seetesh Pande and Tejass Mohan Ukarde and Amanda Gollo Bertollo and Mahesh Joshi and Kihwan Han},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/datasets/TuringEnterprises/Open-RL}
}
```
