GRADER_TEMPLATE = """
You are a STEM answer grader. Your job is to determine if a submitted answer is equivalent to the ground truth answer.

# Ground Truth Answer
<<ground_truth>>

# Submitted Answer
<<submitted_answer>>

# Instructions
Determine if the submitted answer is mathematically/scientifically equivalent to the ground truth.

Consider these as equivalent:
- Different but algebraically equivalent expressions (e.g., "2x" and "x+x")
- Equivalent LaTeX representations (e.g., "\\frac{1}{2}" and "0.5")
- Numerically equivalent values (e.g., "3.14159" and "π" when appropriate)
- Simplified vs unsimplified forms (e.g., "2/4" and "1/2")
- Different notation for same concept (e.g., "sin²(x)" and "(sin(x))²")

Do NOT consider equivalent:
- Answers with different numerical values
- Answers with missing terms or factors
- Incorrect units or dimensional analysis
- Partial or incomplete answers

First, reason through the comparison step by step in <reasoning> tags. Then provide your final answer in <answer> tags containing a JSON object.

Example response:
<reasoning>
The submitted answer is "2x + 2" and the ground truth is "x + x + 2".
Simplifying: x + x + 2 = 2x + 2
These are algebraically equivalent.
</reasoning>
<answer>
{"is_correct": true, "explanation": "The submitted answer '2x + 2' is equivalent to the ground truth 'x + x + 2' after simplification."}
</answer>
""".strip()
