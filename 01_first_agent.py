'''
Writer Agent --> access website search tool --> access some information --> blog post

'''

import os
from dotenv import load_dotenv
load_dotenv(override=True)

from crewai import Agent, Task, Crew
from crewai_tools import WebsiteSearchTool

# create the tool object
search_tool = WebsiteSearchTool()

# define the agent
writer_agent = Agent(
    role = "Blog writer agent",
    goal = "Write short clear blog posts for social media based on the latest information from the web.",
    backstory = "You can explain complex topics with ease and simple analogies",
    llm = "gpt-4o",
    tools = [search_tool] # agent is using search tool to perform its task
)

# assigning the task to the agent
write_task = Task(
    description ="Write a 350 word post about the {topic}",
    expected_output ="A blog post for LinkedIn , with clickable title , 4 paragraphs",
    agent = writer_agent # assigning the agent to the task
)

# assembling the crew with the agent and the task
crew = Crew(
    agents = [writer_agent],
    tasks = [write_task],
    verbose = True
)

# execute 
result = crew.kickoff(inputs= {"topic" : "Recent launches by NVIDIA in GPU as of 2026  "})

print(result)