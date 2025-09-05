import os
from mcp.server.fastmcp import FastMCP
from typing import List
from pathlib import Path

TASK_FILE = "tasks.txt"

mcp = FastMCP("TaskManagementAssistant")

@mcp.tool()
def add_task(task_description: str) -> str:
    try:
        with open(TASK_FILE, "a", encoding="utf-8") as f:
            f.write(task_description.strip() + "\n")
        return f"Task added: {task_description}"
    except Exception as e:
        return f"Error adding task: {e}"
    
@mcp.tool()
def list_tasks() -> List[str]:
    if not os.path.exists(TASK_FILE):
        return ["No tasks found."]
    try:
        with open(TASK_FILE, "r", encoding="utf-8") as f:
            tasks = [line.strip() for line in f.readlines()]
        
        return [task for task in tasks if task]
    except Exception as e:
        print(f"Error reading tasks: {e}")
        return []
    
@mcp.prompt()
def plan_trip_prompt(destination: str, duration_in_days: int) -> str:
    return f"""
    You are an expert travel consultant. Your goal is to help the user by generating a sample travel itinerary and saving it to their task list for later reference.

    The user wants a plan for a {duration_in_days}-day trip to {destination}.

    Follow these steps carefully:
    1.  First, use your general knowledge to brainstorm a simple, day-by-day itinerary. Suggest one or two key attractions or activities for each day of the trip.
    2.  After you have formulated the plan, you MUST perform a critical action: for each individual activity or attraction in your suggested itinerary, save it to the user's task list. For example, if you suggest visiting the Louvre, you must call the tool for that specific item.
    3.  Once all the itinerary items have been added as tasks, present a friendly confirmation message to the user. Inform them that you have created a sample plan and saved it to their to-do list.
    """

@mcp.resource("file://meeting_notes")
def meeting_notes_resource() -> list[str]:
    try:
        notes_file = Path("meeting_notes.txt")
        if not notes_file.exists():
            return ["Error: File not found on the server"]
        return notes_file.read_text(encoding="utf-8").strip().splitlines()
    
    except Exception as e:
        return [f"Error reading meeting notes: {e}"]


if __name__ == "__main__":
    print("Starting Task Management MCP Server...")
    mcp.run(transport="stdio")
