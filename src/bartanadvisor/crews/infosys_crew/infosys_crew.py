from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Optional, Any
import yaml
from pathlib import Path
# from tartanadvisor.src.tartanadvisor.faiss_store import tools
from bartanadvisor.tools.custom_tool import SearchAdvisingBATool, SearchAdvisingISTool, SearchAdvisingCSTool, SearchAdvisingBioTool, SearchCoursesTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class InfosysCrew():
    """InfosysCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def information_systems_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['information_systems_advisor'], # type: ignore[index]
            tools = [SearchAdvisingISTool()],
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
        """Creates the InfosysCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        # manager = Agent(
        #     config=self.agents_config['project_manager'],
        #     allow_delegation=True,
        # )
        print('start2')
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            # manager_agent=manager,  # Use your custom manager agent
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
            # planning=True,
            process=Process.sequential,
            verbose=True
        )
