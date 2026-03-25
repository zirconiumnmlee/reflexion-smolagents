from smolagents import ReflexionAgent, OpenAIModel, tool


@tool
def get_weather(location: str, celsius: bool | None = False) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"


# Configure your model here
model = OpenAIModel(model_id="gpt-4o-mini")

agent = ReflexionAgent(tools=[get_weather], model=model, verbosity_level=2, trials=2)

print("ReflexionAgent:", agent.run("What's the weather like in Paris?"))