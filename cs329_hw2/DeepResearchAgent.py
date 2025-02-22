import os
import requests
from datetime import datetime
import openai
import json
from googleapiclient.discovery import build
from textblob import TextBlob
from openai import OpenAI
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple
from pprint import pprint

from cs329_hw2.api_manager import APIManager
from cs329_hw2.utils import generate_openai, generate_anthropic, generate_together
from dotenv import load_dotenv, find_dotenv
import json

load_dotenv(find_dotenv())

oaiclient = OpenAI()

sonar_client = OpenAI(api_key=os.getenv("SONAR_API_KEY"),
                      base_url=os.getenv("SONAR_BASE_URL"))

class SectionOutline(BaseModel):
    section_name: str
    description: str

class ReportStructure(BaseModel):
    reasoning: str 
    outline: List[SectionOutline]

def get_report_generation_prompt(query: str, schema: BaseModel) -> str:
    return f"""
You are an AI assistant tasked with creating a structured outline for a research report based on a given query. Your goal is to analyze the query and generate a comprehensive outline that will guide the creation of a detailed report addressing all aspects of the user's request.

Here are your instructions:

1. Carefully read and analyze the following query:
<query>
{query}
</query>

2. Consider the main topics, subtopics, and potential areas of investigation suggested by the query. Think about what information would be most useful and relevant to the user based on their request.

3. Create a logical structure for the report outline. This should include main sections and subsections that cover all aspects of the query comprehensively.

4. For each section and subsection, create a clear and concise name that reflects its content. Then, write a brief description of what information should be included in that section.

5. Organize your outline as a JSON-like dictionary structure, where the keys are the section names and the values are the descriptions of what should be included in each section.

6. Use nested dictionaries for subsections. Main sections should be at the top level of the dictionary, with subsections nested within their respective main sections.

7. Ensure that your outline covers all aspects of the query and provides a comprehensive structure for the final report.

8. Remember it should not be over verbose, it should be a high quality concise outline that makes sure everything in the query and pertinent to the query is covered.

Once done reasonging, and creating the outline, retun in the following JSON-like dictionary format:

{schema.model_json_schema()}

Now, based on the query provided, create a comprehensive outline for the research report. Present your outline in the JSON-like dictionary format described above. Ensure that your outline is thorough, logical, and directly addresses all aspects of the query.

    """



class SectionQuestions(BaseModel):
    questions: List[str]


def get_question_generation_prompt(query: str, report_outline: str, section: SectionOutline, schema: BaseModel) -> str:
    return f"""You are an AI assistant tasked with generating relevant research questions for a specific section of a report outline. Your goal is to create 5 questions that, when answered, will provide the necessary information to write a comprehensive and informative section of the report.

First, carefully read and analyze the following query that was used to generate the report outline:

<query>
{query}
</query>

Now, review the structure of the report outline that was created based on this query:

<report_outline>
{report_outline}
</report_outline>

You will be focusing on generating questions for the following section of the report:

<section>
Section Name: {section.section_name}
Section Description: {section.description}
</section>

Your task is to generate 5 specific, focused questions that will guide research for this section. These questions should:

1. Be directly relevant to the content described in the section
2. Cover different aspects or subtopics within the section
3. Be specific enough to guide targeted research
4. Be open-ended enough to encourage comprehensive answers
5. Help gather information that will contribute to a thorough and informative section of the report


Return the questions as a list of strings, in the following format:

{schema.model_json_schema()}

Now, based on the query, report outline, and specified section, generate 5 research questions following the format described above. Ensure that your questions are thorough, relevant, and will effectively guide the research needed to write a comprehensive section of the report. Make sure only specifically returning the to the point questions that should be searched for, not any other text.
"""


def generate_final_report_prompt(query: str, report_outline: str, synthesized_answers: str, citations: List[str]) -> str:
    return f"""
You are an AI assistant tasked with writing a comprehensive final report based on deep research conducted on a complex query. Your goal is to synthesize all the information gathered and present it in a coherent, well-structured manner that directly addresses the initial query.

Here is the initial query that prompted this research:

<query>
{query}
</query>

The research has been conducted based on the following report structure:

<report_structure>
{report_outline}
</report_structure>

For each section of the report, extensive research has been conducted, and answers to specific questions have been synthesized. Here are the synthesized answers for each section:

<synthesized_answers>
{synthesized_answers}
</synthesized_answers>

Your task is to write a final report that combines all this information. Follow these guidelines:

1. Start with an introduction that restates the initial query and provides an overview of the report's structure.

2. For each section in the report structure:
   a. Begin with a clear heading matching the section name.
   b. Synthesize the information from the corresponding part of the synthesized answers.
   c. Ensure smooth transitions between subsections and main ideas.
   d. Integrate relevant information from other sections if it adds value or provides context.

3. Throughout the report:
   a. Maintain a logical flow of ideas.
   b. Highlight key findings and insights.
   c. Draw connections between different sections when relevant.
   d. Provide analysis and interpretation of the information, not just a summary.

4. Conclude the report with:
   a. A summary of the main findings.
   b. Direct answers to the initial query, if not already explicitly stated.
   c. Implications of the findings or areas for further research, if applicable.

5. There are some citation numbers in the synthesized answers, you can get rid of them, we are processing them separately.


Ensure that you:
- Directly address all aspects of the initial query.
- Provide a comprehensive and cohesive narrative that goes beyond simply restating the synthesized answers.
- Use clear, professional language appropriate for a research report.
- Maintain factual accuracy based on the provided information.
- Properly attribute information using the provided citations.

The report should be well-formatted, with clear headings for each section and appropriate paragraph breaks.
"""
class DeepResearchAgent:
    def __init__(self, api_manager):
        self.api_manager = api_manager
        
    def research(self, query: str) -> Dict[str, Any]:
        """
        Conducts deep research on a given query.
        
        Args:
            query: Complex research question to investigate
            
        Returns:
            Dictionary containing:
            - report: str, synthesized findings with citations
            - sources: List[str], list of source URLs or references
        """
        try:
            all_citations = []
            ### Create plan for sections
            report_generation_prompt = get_report_generation_prompt(query, ReportStructure)
            report_structure = self.call_openai_structured(report_generation_prompt, ReportStructure)
            
            report_outline = ""
            for section in report_structure.outline:
                report_outline += f"{section.section_name}: {section.description}\n"
            
            ### Generate questions for each section
            section_questions_map = {}  # Change to dictionary with section name as key
            for section in report_structure.outline:
                question_generation_prompt = get_question_generation_prompt(query, report_outline, section, SectionQuestions)
                questions = self.call_openai_structured(question_generation_prompt, SectionQuestions)
                section_questions_map[section.section_name] = questions.questions  # Store questions list for each section
            
            ### Search and synthesize answers
            synthesized_answers = ''
            
            for section in report_structure.outline:
                synthesized_answers += f"\n## {section.section_name}\n{section.description}\n\n"
                
                # Get questions for this section from our map
                questions = section_questions_map[section.section_name]
                for question in questions:
                    search_results, citations = self.call_sonar(question)
                    synthesized_answers += f"Q: {question}\nA: {search_results}\n\n"
                    all_citations.extend(citations)
            
            # generate the final report 
            report_prompt = generate_final_report_prompt(query, report_outline, synthesized_answers, all_citations)
            final_report = self.call_openai(report_prompt)
            
            return {
                "report": final_report,
                "sources": all_citations
            }
        except Exception as e:
            return {
                "report": f"Error occurred during research: {str(e)}",
                "sources": []
            }
            
    def call_sonar(self, query: str) -> Tuple[str, List[str]]:
        messages = [
            {"role": "system", "content": "You are a deep research agent. You are given a research question and you need to answer it by searching the web and synthesizing the findings."},
            {"role": "user", "content": query}
        ]
        response = sonar_client.chat.completions.create(
            model="sonar",
            messages=messages
        )
        return response.choices[0].message.content, response.citations
    
    
    def call_openai(self, query: str) -> str:
        messages = [
            {"role": "user", "content": query}
        ]
        response = oaiclient.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0
            )
        
        return response.choices[0].message.content
    
    
    def call_openai_structured(self, query: str, schema: BaseModel) -> Dict[str, Any]:
        messages = [
            {"role": "user", "content": query}
        ]
        
        response = oaiclient.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=schema,
            temperature=0.0
        )
        
        return response.choices[0].message.parsed