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
from langchain_mcp_adapters.client import MultiServerMCPClient

# server_configs = {
#     "weather": {
#         "command": "python",
#         "args": ["weather_server.py"],
#         "transport": "stdio"
#     },
#     "tasks": {
#         "command": "python",
#         "args": ["task_server.py"],
#         "transport": "stdio"
#     }
# }

server_params = StdioServerParameters(
    command="python",
    args=["rag_server.py"]
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
        ("system", "You are a helpful RAG assistant. Your role is to answer questions using the content of documents provided by the user. When a user gives you a file path, use your tool to ingest it into your memory. When they ask a question, use your search tool to find the relevant context within the ingested documents and use that context to form a clear answer."),
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

# async def list_prompts(session):
#     try:
#         prompt_response = await session.list_prompts()
#         if not prompt_response or not prompt_response.prompts:
#             print("No prompts were found on the server.")
#             return

#         for p in prompt_response.prompts:
#             print(f"Prompt: {p.name}")
#             if p.arguments:
#                 arg_list = [f"<{arg.name}> " for arg in p.arguments]
#                 print(f"Arguments: {''.join(arg_list)}")
#             else:
#                 print("No arguments found.")

#     except Exception as e:
#         print(f"Error listing prompts: {e}")

# async def handle_prompts(session, command: str) -> str | None:

#     try:
#         parts = shlex.split(command.strip())
#         if len(parts) < 2:
#             print("Usage: /prompt <prompt_name> [arg1=value1 arg2=value2 ...]")
#             return None
        
#         prompt_name = parts[1]
#         user_args = parts[:2]

#         prompt_def_response = await session.list_prompts()
#         if not prompt_def_response or not prompt_def_response.prompts:
#             print("\nError: Could not retrieve any prompts from the server.")
#             return None
        
#         prompt_def = next((p for p in prompt_def_response.prompts if p.name == prompt_name), None)
#         if not prompt_def:
#             print(f"\nError: Prompt '{prompt_name}' not found.")
#             return None

#         if len(user_args) != len(prompt_def.arguments):
#             expected_args = [arg.name for arg in prompt_def.arguments]
#             print(f"\nError: Invalid number of arguments for prompt '{prompt_name}'.")
#             print(f"Expected {len(expected_args)} arguments: {', '.join(expected_args)}")
#             return None
        
#         arg_dict = {arg.name: val for arg, val in zip(prompt_def.arguments, user_args)}

#         prompt_response = await session.get_prompt(prompt_name, arg_dict)

#         prompt_text = prompt_response.messages[0].content.text

#         return prompt_text
        
#     except Exception as e:
#         print(f"Error parsing command: {e}")
#         return None

# #resources
# async def list_resources(session):
#     try:
#         resource_response = await session.list_resources()
#         if not resource_response or not resource_response.resources:
#             print("No resources were found on the server.")
#             return

#         for r in resource_response.resources:
#             print(f"Resource: {r.uri}")
#             if r.description:
#                 print(f"Description: {r.description.strip()}")

#     except Exception as e:
#         print(f"Error listing resources: {e}")

# async def handle_resources(session, command: str) -> str | None:
#     try:
#         parts = shlex.split(command.strip())
#         if len(parts) != 2:
#             print("Usage: /resource <resource_name>")
#             return None

#         resource_uri = parts[1]

#         response = await session.read_resource(resource_uri)
#         if not response or not response.contents:
#             print(f"Error: Resource '{resource_uri}' not found.")
#             return None
        
#         text_parts = [
#             content.text for content in response.contents if hasattr(content, "text")
#         ]

#         if not text_parts:
#              print("Error: Resource content is not in a readable text format.")
#              return None
        
#         resource_content = "\n".join(text_parts)

#         return resource_content

#     except Exception as e:
#         print(f"Error handling resources: {e}")
#         return None

# async def list_all_prompts(client: MultiServerMCPClient, server_configs: dict): 
#     print("\nAvailable Prompts from all servers:")
#     print("-----------------------------------")
    
#     any_prompts_found = False

#     for server_name in server_configs.keys():
#         try:
#             async with client.session(server_name) as session:
#                 prompt_response = await session.list_prompts()

#                 if prompt_response and prompt_response.prompts:
#                     any_prompts_found = True
#                     print(f"\nServer: {server_name}")
#                     for p in prompt_response.prompts:
#                         print(f"  Prompt: {p.name}")
#                         if p.arguments:
#                             arg_list = [arg.name for arg in p.arguments]
#                             print(f"    Arguments: {', '.join(arg_list)}")
#                         else:
#                             print("    Arguments: None")
#         except Exception as e:
#             print(f"Error listing prompts from server '{server_name}': {e}")
    
#         print("\nUse: /prompt <server_name> <prompt_name> \"arg1\" \"arg2\" ...")
#         print("-----------------------------------")
#         if not any_prompts_found:
#             print("\nNo prompts were found on any connected servers.")

# async def handle_prompt_invocation(client: MultiServerMCPClient, command: str) -> str | None:
#     try:
#         parts = shlex.split(command.strip())
#         if len(parts) < 3:
#             print("Usage: /prompt <server_name> <prompt_name> [arg1 arg2 ...]")
#             return None
        
#         server_name = parts[1]
#         prompt_name = parts[2]
#         user_args = parts[3:]

#         if server_name not in server_configs:
#             print(f"\nError: Server '{server_name}' is not recognized.")
#             return None

#         prompt_def = None
#         async with client.session(server_name) as session:
#             prompt_def_response = await session.list_prompts()
#             if not prompt_def_response or not prompt_def_response.prompts:
#                 print(f"\nError: Could not retrieve any prompts from the server '{server_name}'.")
#                 return None
            
#             prompt_def = next((p for p in prompt_def_response.prompts if p.name == prompt_name), None)
#             if not prompt_def:
#                 print(f"\nError: Prompt '{prompt_name}' not found on server '{server_name}'.")
#                 return None

#             if len(user_args) != len(prompt_def.arguments):
#                 expected_args = [arg.name for arg in prompt_def.arguments]
#                 print(f"\nError: Invalid number of arguments for prompt '{prompt_name}'.")
#                 print(f"Expected {len(expected_args)} arguments: {', '.join(expected_args)}")
#                 return None
            
#             arg_dict = {arg.name: val for arg, val in zip(prompt_def.arguments, user_args)}

#             prompt_messages = await client.get_prompt(
#                 server_name=server_name,
#                 prompt_name=prompt_name,
#                 arguments=arg_dict
#             )
#             print("\n--- Prompt loaded successfully. Preparing to execute... ---")
#             prompt_text = prompt_messages[0].content

#             return prompt_text
        
#     except Exception as e:
#             print(f"Error handling prompt invocation: {e}")
#             return None

# async def list_all_resources(client: MultiServerMCPClient, server_configs: dict):
#     print("\nAvailable Resources from all servers:")
#     print("-------------------------------------")

#     any_resources_found = False

#     for server_name in server_configs.keys():
#         try:
#             async with client.session(server_name) as session:
#                 resource_response = await session.list_resources()
#                 if resource_response and resource_response.resources:
#                     any_resources_found = True
#                     print(f"\n--- Server: '{server_name}' ---")
#                     for r in resource_response.resources:
#                         print(f"  Resource URI: {r.uri}")
#                         if r.description:
#                             print(f"    Description: {r.description}")

#         except Exception as e:
#             print(f"Error listing resources from server '{server_name}': {e}")

#     print("\nUse: /resource <server_name> <resource_uri>")
#     print("-----------------------------------")     

#     if not any_resources_found:
#         print("\nNo resources were found on any connected servers.")

# async def handle_resource_invocation(client: MultiServerMCPClient, command: str) -> str | None:
#     try:
#         parts = shlex.split(command.strip())
#         if len(parts) != 3:
#             print("Usage: /resource <server_name> <resource_uri>")
#             return None

#         server_name = parts[1]
#         resource_uri = parts[2]

#         print(f"\n--- Fetching resource '{resource_uri}' from server '{server_name}'... ---")

#         blobs = await client.get_resources(server_name=server_name, uris=[resource_uri])

#         if not blobs:
#             print("Error: Resource not found.")
#             return None

#         resource_content = blobs[0].as_string()

#         return resource_content
    
#     except Exception as e:
#         print(f"Error handling resource invocation: {e}")
#         return None

# entry point
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
    
            agent = create_graph(session)

            print("""
            Hello! I'm your document assistant.

            To get started, tell me which document to read by providing its path, like:
                Example: ingest_document /usercode/Guides/employee_handbook.txt

            Once it's loaded, I can answer any questions you have about it!
            """)

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    break

                try:
                    response = await agent.invoke(
                        {"messages": user_input},
                        # ---  Thread ID for clarity ---
                        config={"configurable": {"thread_id": "weather-session"}}
                    )
                    print("AI:", response["messages"][-1].content)
                except Exception as e:
                    print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())