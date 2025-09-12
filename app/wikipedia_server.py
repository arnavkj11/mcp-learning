import wikipedia 
from typing import List, Dict, Any
import logging
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WikipediaSearch")

@mcp.tool()
def search_wikipedia(query: str, num_results: int = 1) -> List[Dict[str, Any]]:
    """
    Searches Wikipedia for a given query and returns a list of summaries for the top results.
    """
    try:
        results = wikipedia.search(query, results=num_results)
        if not results:
            return [{"error": f"No wikipedia results found for query: {query}"}]
        
        summaries = []
        for title in results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summaries.append({
                    "title": page.title,
                    "summary": page.summary.split('\n')[0],  # Return only the first paragraph
                    "url": page.url
                })
            except Exception as e:
                logging.error("Error fetching Wikipedia page for title '%s': %s", title, e)
        return summaries
    except Exception as e:
        logging.error("Error searching Wikipedia: %s", e)
        return []
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    mcp.run(transport="stdio")