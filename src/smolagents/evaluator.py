import json
from typing import Any

import yaml
import importlib.resources

from .models import ChatMessage, MessageRole


def format_trajectory(trajectory: list[dict]) -> str:
    """Format the execution trajectory into a readable string."""
    formatted_steps = []
    for i, step in enumerate(trajectory):
        step_num = step.get("step_number", i + 1)
        tool_calls = step.get("tool_calls", [])
        observations = step.get("observations", "")
        error = step.get("error")
        is_final = step.get("is_final_answer", False)
        model_output = step.get("model_output", "")

        step_str = f"Step {step_num}:\n"
        step_str += f"  Model Output: {model_output}\n"

        if tool_calls:
            step_str += f"  Tool Calls:\n"
            for tc in tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("arguments", {})
                step_str += f"    - {name}({args})\n"

        if observations:
            step_str += f"  Observation: {observations}\n"

        if error:
            step_str += f"  Error: {error}\n"

        if is_final:
            step_str += f"  [FINAL ANSWER]\n"

        formatted_steps.append(step_str)

    return "\n".join(formatted_steps)


def evaluator(
    trajectory: list[dict],
    model: Any,
    task: str = "",
    prompt_path: str = "smolagents.prompts.reflexion_agent.yaml",
) -> tuple[bool, float]:
    """
    Evaluate the agent's execution trajectory using a language model.

    Args:
        trajectory: List of step dictionaries from the agent's execution.
        model: The language model to use for evaluation.
        task: The original task description.
        prompt_path: Path to the prompt template.

    Returns:
        A tuple of (is_pass: bool, reward: float)
    """
    # Load the prompt template
    prompt_template = yaml.safe_load(
        importlib.resources.files("smolagents.prompts")
        .joinpath("reflexion_agent.yaml")
        .read_text()
    )

    evaluator_prompt_template = prompt_template.get("evaluator_prompt", "")

    # Format the trajectory
    formatted_trajectory = format_trajectory(trajectory)

    # Populate the prompt
    prompt = evaluator_prompt_template.format(
        trajectory=formatted_trajectory,
        task=task,
    )

    # Call the model to evaluate
    messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])]

    response = model.generate(messages)

    # Parse the response
    response_text = response.content[0]["text"] if isinstance(response.content, list) else response.content

    # Extract JSON from the response
    try:
        # Try to find JSON in the response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
            is_pass = result.get("is_pass", False)
            reward = float(result.get("reward", 0.0))
        else:
            # Fallback: try to parse the whole response as JSON
            result = json.loads(response_text)
            is_pass = result.get("is_pass", False)
            reward = float(result.get("reward", 0.0))
    except (json.JSONDecodeError, ValueError):
        # If JSON parsing fails, default to failed
        is_pass = False
        reward = 0.0

    return is_pass, reward