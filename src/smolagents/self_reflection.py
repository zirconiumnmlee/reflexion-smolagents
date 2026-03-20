"""Self-reflection for ReflexionAgent - generates insights from execution trajectories."""

import yaml
import importlib.resources

from .evaluator import format_trajectory


def self_reflection(
    trajectory: list[dict],
    reward: float,
    model,
    task: str = "",
    prompt_path: str = "smolagents.prompts.reflexion_agent.yaml",
) -> str:
    """
    Generate self-reflection insights from the agent's execution trajectory.

    Args:
        trajectory: List of step dictionaries from the agent's execution.
        reward: The reward score from the evaluator.
        model: The language model to use for reflection.
        task: The original task description.
        prompt_path: Path to the prompt template.

    Returns:
        A string containing self-reflection insights.
    """
    # Load the prompt template
    prompt_template = yaml.safe_load(
        importlib.resources.files("smolagents.prompts")
        .joinpath("reflexion_agent.yaml")
        .read_text()
    )

    reflection_prompt_template = prompt_template.get("self_reflection_prompt", "")

    # Format the trajectory
    formatted_trajectory = format_trajectory(trajectory)

    # Determine if passed based on reward
    is_pass = reward >= 0.5

    # Populate the prompt
    prompt = reflection_prompt_template.format(
        trajectory=formatted_trajectory,
        task=task,
        is_pass=is_pass,
        reward=reward,
    )

    # Call the model to generate reflection
    from .models import ChatMessage, MessageRole

    messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])]

    response = model.generate(messages)

    # Extract the reflection text
    reflection = (
        response.content[0]["text"]
        if isinstance(response.content, list)
        else response.content
    )

    return reflection