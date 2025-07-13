#!/usr/bin/env python

import os

from dotenv import load_dotenv
from tartanadvisor.crews.bio_crew.bio_crew import BioCrew
from tartanadvisor.crews.business_crew.business_crew import BusinessCrew
from tartanadvisor.crews.compscience_crew.compscience_crew import CompscienceCrew
load_dotenv()

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router

from crewai import LLM

import yaml

from tartanadvisor.crews.infosys_crew.infosys_crew import InfosysCrew

from tartanadvisor.faiss_store import STORES

llm = LLM(model = os.getenv("MODEL_REASONING"))

class directionState(BaseModel):
    user_message: str = ""
    direction: str = ""
    answer: str = ""

class AdvisingFlow(Flow[directionState]):

    @start()
    def get_user_input(self):
        user_message = input("What do you need advising with? Choose from: Business Administration, Information Systems, Computer Science, and Biological Sciences. ")
        while True:
            prompt = (
            "Depending on the user input, deduct if they want academic advising or not. "
            "By ‘advising’ we mean course details, concentrations, policies, requirements, sample curricula, etc. "
            "If it’s about academic advising, return exactly one of: "
            "Business Administration, Information Systems, Computer Science, Biological Sciences. "
            "If it’s advising about a major or a minor or concentration, but not about Business Administration, Information Systems, Computer Science, or Biological Sciences return 'Not-defined'. "
            "If it’s not academic advising at all, return 'Not-Advising'.\n\n"
            "If it's something completely else, then return 'Unknown'"
            f"Here is the user input: {user_message}"
            )
            print(prompt)
            intention = llm.call(prompt)
            print("⟵ raw LLM output:", intention)
            if not ("Not-Advising" in intention or "Not-defined" in intention or "Not-defined" in intention):
                break
            else:
                if ("Not-Advising" in intention):
                    user_message = input("This is not an advising question, please ask me an advising related question. ")
                elif ("Not-defined" in intention):
                    user_message = input("I do not currently offer advising in this sphere, please ask me a question related to Business Administration, Information Systems, Computer Science, or Biological Sciences ")
                elif ("Not-defined" in  intention):
                    user_message = input("This is not an advising question, please ask me an advising related question. ")
        self.state.direction = intention.strip()
        print(user_message)
        self.state.user_message= user_message

    @router(get_user_input)
    def direction(self, previous_result):
        if self.state.direction == 'Business Administration':
            return 'Business Administration'
        elif self.state.direction == 'Computer Science':
            return 'Computer Science'
        elif self.state.direction == 'Information Systems':
            return 'Information Systems'
        elif self.state.direction == 'Biological Sciences':
            return 'Biological Sciences'
    
    @listen('Business Administration')
    def business_advisor(self):
        print('starting')
        result = BusinessCrew().crew().kickoff(inputs={
                "topic": self.state.user_message
            })
        print(f"here is the {result}")
        self.state.answer = result.raw
        return result.raw
    
    @listen('Computer Science')
    def computer_advisor(self):
        print('starting')
        result = CompscienceCrew().crew().kickoff(inputs={
                "topic": self.state.user_message
            })
        print(f"here is the {result}")
        self.state.answer = result.raw
        return result.raw
    
    @listen('Information Systems')
    def information_advisor(self):
        print('starting')
        result = InfosysCrew().crew().kickoff(inputs={
                "topic": self.state.user_message
            })
        print(f"here is the {result}")
        self.state.answer = result.raw
        return result.raw
    
    @listen('Biological Sciences')
    def bio_advisor(self):
        print('starting')
        result = BioCrew().crew().kickoff(inputs={
                "topic": self.state.user_message
            })
        print(f"here is the {result}")
        self.state.answer = result.raw
        return result.raw

    @listen(business_advisor)
    def save_answer_business_administration(self):
        print("Saving answer")
        with open("answer7.txt", "w") as f:
            f.write(self.state.answer)

    @listen(computer_advisor)
    def save_answer_computer_science(self):
        print("Saving answer")
        with open("answer6.txt", "w") as f:
            f.write(self.state.answer)

    @listen(information_advisor)
    def save_answer_information_systems(self):
        print("Saving answer")
        with open("answer8.txt", "w") as f:
            f.write(self.state.answer)

    @listen(bio_advisor)
    def save_answer_biological_sciences(self):
        print("Saving answer")
        with open("answer.txt", "w") as f:
            f.write(self.state.answer)

def kickoff():
    advising_flow = AdvisingFlow()
    advising_flow.kickoff()

def plot():
    advising_flow = AdvisingFlow()
    advising_flow.plot()

if __name__ == "__main__":
    kickoff()
