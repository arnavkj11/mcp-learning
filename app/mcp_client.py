import gradio as gr
import os
from dotenv import load_dotenv
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

server_configs = {
    "wikipedia" : {
        "command" : "python",
        "args" : ["wikipedia_server.py"],
        "transport" : "stdio"
    },
    "vision" : {
        "command" : "python",
        "args" : ["VisualAnalysisServer.py"],
        "transport" : "stdio"
    }
}

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

def create_graph(tools: list):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    llm_with_tools = llm.bind_tools(tools)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant. Your purpose is to provide comprehensive answers to user requests. "
               "You have access to a specialized set of tools for analyzing the content of images and another set for researching topics on Wikipedia. "
               "Intelligently chain these tools together to fulfill the user's request. For example, if a user asks about an image, first analyze the image to understand what it is, then use that understanding to perform research."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    def chat_node(state: State) -> State:
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}
    
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

async def main():
    client = MultiServerMCPClient(server_configs)
    all_tools = await client.get_tools()
    agent = create_graph(all_tools)

    print("The Image Research Assistant is ready and launching on a web UI...")

    def get_agent_response(user_text, image_path, chat_history):
        # Prepare the message content
        if image_path:
            full_message = f"{user_text} [Image uploaded: {image_path}]"
            # For chat history, add image and text separately
            chat_history.append((user_text, None))
        else:
            full_message = user_text
            # For chat history, add just the text
            chat_history.append((user_text, None))
        
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            response = loop.run_until_complete(agent.ainvoke(
                {"messages": [{"role": "user", "content": full_message}]},
                config={"configurable": {"thread_id": "gradio-session"}}
            ))

            bot_message = response["messages"][-1].content
            chat_history.append((None, bot_message))
        except (KeyError, IndexError) as e:
            error_message = f"Response format error: {str(e)}"
            chat_history.append((None, error_message))
        except (ConnectionError, TimeoutError) as e:
            error_message = f"Connection error: {str(e)}"
            chat_history.append((None, error_message))
        except ValueError as e:
            error_message = f"Value error: {str(e)}"
            chat_history.append((None, error_message))
        finally:
            if 'loop' in locals():
                loop.close()

        return "", chat_history, None

    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
        gr.Markdown("# Image Research Assistant")
        chatbot = gr.Chatbot(label="Conversation", height=500)

        with gr.Row():
            image_box = gr.Image(type="filepath", label="Upload an image to analyze")

            text_box = gr.Textbox(
                label="Ask a question about the image or a general research question",
                scale=2
            )
        submit_btn = gr.Button("Submit", variant="primary")
        
        submit_btn.click(
            fn=get_agent_response,
            inputs=[text_box, image_box, chatbot],
            outputs=[text_box, chatbot, image_box]
        )

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

if __name__ == "__main__":
    asyncio.run(main())

