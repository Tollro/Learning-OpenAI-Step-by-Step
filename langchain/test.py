from langchain.agents import create_agent

agent = create_agent(
    tools=["search", "calculator"],
    llm="gpt-4o",
)