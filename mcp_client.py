import asyncio
import shlex
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

async def list_prompts(session):
    try:
        prompt_response = await session.list_prompts()
        if not prompt_response or not prompt_response.prompts:
            print("No prompts were found on the server.")
            return

        for p in prompt_response.prompts:
            print(f"Prompt: {p.name}")
            if p.arguments:
                arg_list = [f"<{arg.name}> " for arg in p.arguments]
                print(f"Arguments: {''.join(arg_list)}")
            else:
                print("No arguments found.")

    except Exception as e:
        print(f"Error listing prompts: {e}")

async def handle_prompts(session, command: str) -> str | None:

    try:
        parts = shlex.split(command.strip())
        if len(parts) < 2:
            print("Usage: /prompt <prompt_name> [arg1=value1 arg2=value2 ...]")
            return None
        
        prompt_name = parts[1]
        user_args = parts[:2]

        prompt_def_response = await session.list_prompts()
        if not prompt_def_response or not prompt_def_response.prompts:
            print("\nError: Could not retrieve any prompts from the server.")
            return None
        
        prompt_def = next((p for p in prompt_def_response.prompts if p.name == prompt_name), None)
        if not prompt_def:
            print(f"\nError: Prompt '{prompt_name}' not found.")
            return None

        if len(user_args) != len(prompt_def.arguments):
            expected_args = [arg.name for arg in prompt_def.arguments]
            print(f"\nError: Invalid number of arguments for prompt '{prompt_name}'.")
            print(f"Expected {len(expected_args)} arguments: {', '.join(expected_args)}")
            return None
        
        arg_dict = {arg.name: val for arg, val in zip(prompt_def.arguments, user_args)}

        prompt_response = await session.get_prompt(prompt_name, arg_dict)

        prompt_text = prompt_response.messages[0].content.text

        return prompt_text
        
    except Exception as e:
        print(f"Error parsing command: {e}")
        return None


# entry point
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            agent = await create_graph(session)
            print("Weather MCP agent is ready.")
            print("Type a question, or use one of the following commands:")
            print("  /prompts                           - to list available prompts")
            print("  /prompt <prompt_name> \"args\"...  - to run a specific prompt")

            while True:
                message_to_agent = ""
                user_input = input("Enter a location to get the current weather (or 'exit' to quit): ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("Exiting...")
                    break

                if user_input.lower() == "/prompts":
                    await list_prompts(session)
                    continue

                elif user_input.startswith("/prompt"):
                    prompt_text = await handle_prompts(session, user_input)
                    if prompt_text:
                        message_to_agent = prompt_text
                    else:
                        continue
                
                else:
                    message_to_agent = user_input

                if message_to_agent:
                    try:
                        response = await agent.ainvoke(
                            {"messages": [("user", message_to_agent)]},
                            config={"configurable": {"thread_id": "weather-session"}}
                        )
                        print("AI:", response["messages"][-1].content)
                    except Exception as e:
                        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())