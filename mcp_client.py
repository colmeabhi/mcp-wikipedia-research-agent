import asyncio
import os
import shlex
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from langchain_mcp_adapters.tools import load_mcp_tools


# MCP server launch config
server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

# LangGraph state definition
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


async def create_graph(session):
    tools = await load_mcp_tools(session)

    # Get and validate API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set.\n"
            "Please set it using: export GOOGLE_API_KEY='your-api-key'\n"
            "Get your API key from: https://makersuite.google.com/app/apikey"
        )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that uses tools to explore Wikipedia."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())

# Entry point
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = await create_graph(session)

            print("Wikipedia MCP agent is ready.")
            print("Type a question, or 'exit' to quit.")

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    break

                try:
                    response = await agent.ainvoke(
                        {"messages": user_input},
                        config={"configurable": {"thread_id": "wiki-session"}}
                    )
                    print("AI:", response["messages"][-1].content)
                except Exception as e:
                    print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())