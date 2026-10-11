import os
from dotenv import load_dotenv
load_dotenv(override=True)

from crewai import Agent, Task ,Crew


# define the writer agent
writer_agent = Agent(
    role = "Blog writer",
    goal = "Write short , clear blog posts for beginners",
    backstory = "You explain complex technical topics in simple terms \
        with interesting analogies",
    llm = "gpt-4o"
)

# define the task , that I will be giving to the agent
write_task = Task(
    description = "Write a 150 word blog post about the {topic}.", 
    expected_output="A blog post with a title, and 4 short paragraphs, this is for social media",
    agent = writer_agent
)

# define the crew , that will manage the task
crew = Crew(
    agents = [writer_agent],
    tasks = [write_task],
    verbose = True
)

result = crew.kickoff(inputs = {"topic": "importance of AI in education"})

print(result)