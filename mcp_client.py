import asyncio
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables from .env file
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode

from typing import List, Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.tools import load_mcp_tools

server_params = StdioServerParameters(
    command="python",
    args=["weather_server.py"]
)

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

async def create_graph(session):
    tools = await load_mcp_tools(session)

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    llm_with_tools = llm.bind_tools(tools)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that uses tools to get the current weather for a location"),
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

# entry point
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            agent = await create_graph(session)
            print("Weather MCP agent is ready.")

            while True:
                user_input = input("Enter a location to get the current weather (or 'exit' to quit): ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("Exiting...")
                    break

                try:
                    response = await agent.ainvoke(
                        {"messages": user_input},
                        config={"configurable": {"thread_id": "weather_session"}}
                    )
                    print("AI:", response["messages"][-1].content)
                except KeyError as e:
                    print(f"Error accessing response data: {e}")
                except ValueError as e:
                    print(f"Error processing input: {e}")
                except (ConnectionError, TimeoutError) as e:
                    print(f"Connection error: {e}")
                except Exception as e:
                    print(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())