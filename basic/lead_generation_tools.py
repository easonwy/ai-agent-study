# Import the generic tool wrapper from LangChain core
from langchain_core.tools import tool

# Standard libraries for scraping and saving
from datetime import datetime
import requests 
from bs4 import BeautifulSoup
import re
import logging

# Use duckduckgo_search directly to avoid langchain_community version issues
from duckduckgo_search import DDGS

# Set up logging
logger = logging.getLogger(__name__)

@tool
def search_tool_func(query: str) -> str:
    """
    Performs a web search using DuckDuckGo.
    
    Args:
        query: The search query.
        
    Returns:
        A string containing search results with titles, links and snippets.
    """
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found."
        
        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r.get('title', '')}\nLink: {r.get('href', '')}\nSnippet: {r.get('body', '')}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed: {e}"

# Assign the tool to 'search' variable to maintain compatibility
search = search_tool_func

@tool
def save_to_text(data: str, filename: str = "leads_output.txt", append: bool = True) -> str:
    """
    Saves the given content to a text file.
    
    Args:
        data: The content to save
        filename: The name of the file to save to
        append: Whether to append to the file or overwrite it
    
    Returns:
        A message confirming the save operation
    """
    logger.debug(f"Saving data to file: {filename}, append: {append}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Leads Output ---Timestamp: {timestamp}\n\n{data}\n\n"

    # Open the file in append or write mode
    mode = 'a' if append else 'w'
    try:
        with open(filename, mode, encoding='utf-8') as f:
            f.write(formatted_text)
        logger.info(f"Data successfully saved to {filename}")
        return f"Data successfully saved to {filename}"
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return f"Error saving file: {e}"


@tool
def scrape_website(url: str) -> str:
    """
    Scrape raw text from a website.
    
    Args:
        url: The URL of the website to scrape
    
    Returns:
        The scraped text content, or an error message if scraping fails
    """
    logger.debug(f"Scraping website: {url}")
    try:
        # Send GET request to the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        logger.debug(f"Got response from {url}")

        # Parse and clean up the raw HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=" ", strip=True)

        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit to 5000 characters to avoid overloading the model
        logger.debug(f"Scraped {len(text)} characters from {url}")
        return text[:5000]
    
    except requests.RequestException as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return f"Error scraping {url}: {str(e)}"


def generate_search_queries(company_name: str) -> list[str]:
    """
    Generate search queries to look for IT services related to a company.
    
    Args:
        company_name: The name of the company to generate queries for
    
    Returns:
        A list of search queries
    """
    keywords = ["IT Services", "managed IT", "technology solutions"]
    return [f"{company_name} {keyword}" for keyword in keywords]


@tool
def search_and_scrape(company_name: str) -> str:
    """
    Combined search and scrape operation for a company.
    
    Args:
        company_name: The name of the company to search and scrape
    
    Returns:
        Combined text results from search and scraping
    """
    logger.debug(f"Searching and scraping for company: {company_name}")
    queries = generate_search_queries(company_name)
    logger.debug(f"Generated queries: {queries}")
    results = []

    for query in queries:
        try:
            # Run web search
            logger.debug(f"Running search for: {query}")
            # Use search.invoke because it's a Tool now
            search_results = search.invoke(query)
            logger.debug(f"Search completed for: {query}")

            # Extract URLs from the search output
            urls = re.findall(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                search_results
            )
            logger.debug(f"Found URLs: {urls}")

            # Scrape the first valid URL found
            if urls:
                logger.debug(f"Scraping first URL: {urls[0]}")
                # Use scrape_website directly (it's a tool but also a callable function if wrapped properly,
                # actually @tool decorated function is callable?
                # LangChain @tool returns a generic Tool object which is callable via .invoke or .run
                # But the original function is accessible via .func or wrapper logic.
                # However, inside another tool, calling another tool:
                # If `scrape_website` is a StructuredTool, `scrape_website(url)` might fail if not properly typed.
                # Best to use `scrape_website.invoke(url)`.
                results.append(scrape_website.invoke(urls[0]))
        except Exception as e:
            logger.error(f"Error in search_and_scrape: {e}")
            # Continue with other queries if one fails
            continue
        
    # Combine all the text results into one big chunk
    combined_results = " ".join(results)
    logger.debug(f"Combined results length: {len(combined_results)}")
    return combined_results


# Defining all the tools that we created above

search_tool = search
scrape_tool = search_and_scrape
save_tool = save_to_text
