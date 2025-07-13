from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from tartanadvisor.tools.custom_tool import SearchAdvisingBATool, SearchCoursesTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class BusinessCrew():
    """BusinessCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def business_administration_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['business_administration_advisor'], # type: ignore[index]
            tools = [SearchAdvisingBATool()],
            verbose=True
        )
    
    @agent
    def course_finder(self) -> Agent:
        return Agent(
            config=self.agents_config['course_finder'], # type: ignore[index]
            tools = [SearchCoursesTool()],
            verbose=True
        )

    @agent
    def reporter(self) -> Agent:
        return Agent(
            config=self.agents_config['reporter'],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def finding_advising_information(self) -> Task:
        return Task(
            config=self.tasks_config['finding_advising_information'], # type: ignore[index]
        )
    
    @task
    def finding_course(self) -> Task:
        return Task(
            config=self.tasks_config['finding_course'], # type: ignore[index]
        )
    
    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting'],
            context=[self.finding_course(), self.finding_advising_information()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the BusinessCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
