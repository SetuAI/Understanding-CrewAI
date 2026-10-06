import os
from dotenv import load_dotenv
load_dotenv(override=True)

from crewai import Agent, Task, Crew,Process
from crewai_tools import WebsiteSearchTool

WEBSITE = "https://www.oracle.com/in/news/"

# create the tool object
search_tool = WebsiteSearchTool(website=WEBSITE)

# define researcher agent

researcher_agent = Agent(
    role = "Website Researcher agent",
    goal ="Search the website for relevant information on a {topic} from the given website.",
    backstory ="You only report facts that you find on the website. You do not make up information.",
    tools = [search_tool],
    llm = "gpt-4o"
)

# define writer agent
writer_agent = Agent(
    role = "Blog writer agent",
    goal = "Write short clear blog posts for social media based on the latest information from the web.",
    backstory = "You can explain complex topics with ease and simple analogies.\
        You write short and engaging blog posts for social media. You write in a clear and concise manner.",
    llm = "gpt-4o",
)

# define tasks for researcher agent
research_task = Task(
    description="Search the website for relevant information on {topic} from the website.",
    expected_output = "5-7 bullet points of relevant information from the website.",
    agent = researcher_agent
)

# define the write task

write_task = Task(
    description="Write a 350 word post about the {topic} based on the information provided by the researcher agent.",
    expected_output="A 350 word blog post about the {topic}.",
    agent=writer_agent,
    context = [research_task],
    output_file = "blog_post.md"
)
# assembling the crew with the agents and tasks
crew = Crew(
    agents = [researcher_agent, writer_agent],
    tasks = [research_task, write_task],
    process = Process.sequential,
    verbose = True
)

# execute the crew
result = crew.kickoff(inputs = {"topic":"extract news with respect to NTT Docomo"})

print(result)