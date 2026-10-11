'''
2 agents :
1 scrapes the Hacker News front page and extracts relevant stories
2 analyst summarizes the scraped data into a report
'''

import os
from dotenv import load_dotenv
load_dotenv(override=True)
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool

WEBSITE = "https://news.ycombinator.com"

scrape_tool = ScrapeWebsiteTool(website_url = WEBSITE)


# define the scraper agent

scraper_agent = Agent(
    role = "Hacker News scraper",
    goal = "Scrape the Hacker News front page and extract stories related to {topic}",
    backstory = "You are a careful data extractor. You only report what is on the page \
        and you never make up titles, points or comment counts",
    tools = [scrape_tool], # scraper agent has access to the scrape tool
    llm = "gpt-4o",
)


# define the analyst agent

analyst_agent = Agent(
    role = "Tech news analyst",
    goal = "Turn the scraped Hacker News data into a clear, structured report",
    backstory = "You are a tech analyst who spots trends in what developers are talking about \
        and explains them in simple terms",
    llm = "gpt-4o",
)


# define the scrape task

scrape_task = Task(
    description = "Scrape {website} and extract the stories on the front page. \
        For each story capture: rank, title, points, number of comments and author. \
        Mark which stories are related to {topic}",
    expected_output = "A list of front page stories with rank, title, points, comments, author \
        and a flag for whether it relates to {topic}",
    agent = scraper_agent,
)


# define the analysis task

analysis_task = Task(
    description = "Using the scraped data, write a report on the Hacker News front page \
        with a focus on {topic}",
    expected_output = "A markdown report with: a table of the top 10 stories, a section on stories \
        related to {topic}, 3 key trends, and a short conclusion",
    agent = analyst_agent,
    context = [scrape_task],
    output_file = "hn_report.md"
)

# define the linkedin post task (same analyst agent, keeps this a 2 agent crew)

linkedin_task = Task(
    description = "Using the report, write a LinkedIn post about today's Hacker News highlights \
        on {topic}. Plain text only (no markdown tables or headings), hook in the first line, \
        3-5 short bullet points using emojis, a closing question, and 3-5 hashtags. \
        Keep it under 2500 characters",
    expected_output = "A ready-to-publish LinkedIn post in plain text",
    agent = analyst_agent,
    context = [analysis_task],
    output_file = "linkedin_post.txt"
)

# assemble the crew

crew = Crew(
    agents = [scraper_agent, analyst_agent],
    tasks = [scrape_task, analysis_task, linkedin_task],
    process = Process.sequential,
    verbose = True
)


def run_crew(topic = "AI and LLMs"):
    """Runs the crew and returns the LinkedIn post text (used by the Airflow DAG)"""
    result = crew.kickoff(inputs = {"website": WEBSITE, "topic": topic})
    return result.raw


if __name__ == "__main__":
    print(run_crew())
