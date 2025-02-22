import os
import requests
from datetime import datetime
import openai
import json
from googleapiclient.discovery import build
from textblob import TextBlob
from openai import OpenAI
from pydantic import BaseModel
from typing import Dict, Any, List, Literal
from pydantic import ValidationError

from cs329_hw2.api_manager import APIManager
from cs329_hw2.utils import generate_openai, generate_anthropic, generate_together
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

oaiclient = OpenAI()

sonar_client = OpenAI(api_key=os.getenv("SONAR_API_KEY"),
                      base_url=os.getenv("SONAR_BASE_URL"))

class SubQueryParams(BaseModel):
    search_term: str | None = None
    symbol: str | None = None
    text: str | None = None
    location: str | None = None
    date: str | None = None

class SubQuery(BaseModel):
    api: Literal["google_search", "get_stock_data", "analyze_sentiment", "get_weather"]
    params: SubQueryParams
    order: int

class DecompositionResponse(BaseModel):
    sub_queries: List[SubQuery]

class MultiLMAgent:
    """A class to manage multiple language models for generation, iterative refinement, and fusion"""
    
    def __init__(self, api_manager):
        self.api_manager = api_manager

    def generate(self, prompt: str, model: str = "gpt-4o") -> List[Dict]:
        """
        INPUT:
            prompt: str - Input prompt for generation
            model: str - The model to use for generation
        OUTPUT:
            response: str - Generated response
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates responses to user queries."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            if "gpt" in model:
                response = generate_openai(messages=messages, model=model, temperature=0.7)
            elif "claude" in model:
                response = generate_anthropic(messages=messages, model=model, temperature=0.7)
            else:
                response = generate_together(messages=messages, model=model, temperature=0.7)
            
            return response
        except Exception as e:
            return []

    def single_LM_with_single_API_call(self, query: str, model: str) -> str:
        """
        INPUT:
            query: str - The user's query to be decomposed
            model: str - The model to use for generation

        OUTPUT:
            str - The response from the model
        """
        api_retrieved_info = self.api_manager.route_query(query)
        api_response = api_retrieved_info["results"]
        api_used = api_retrieved_info["api_used"]

        prompt = f"""
        You are a helpful assistant that generates responses to user queries.
        Here is the query: {query}
        Here is an api call that was made to retrieve information: {api_used}
        Here is the response from the api call: {api_response}
        """
        
        response = self.generate(prompt, model)
        return response
    
    def get_online_knowledge_using_LLM(self, query: str) -> str:
        """
        INPUT:
            query: str - The user's query to be decomposed
            
        OUTPUT:
            str - Formatted prompt for decomposition model
            list[str] - citations
        """
        
        response = sonar_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        response_text = response.choices[0].message.content
        citations = response.citations
        
        result_string = f"Online Knowledge: {response_text}\nCitations: {citations}"
        
        return result_string
        
        
        
    def decompose_query(self, query: str) -> List[Dict]:
        """
        INPUT:
            query: str - The user's original query to be decomposed
            
        OUTPUT:
            List[Dict] - List of dictionaries containing:
                - api: str - Name of the API to call
                - params: Dict - Parameters for the API call
                - results: Dict - Gathered results from the API call
                - status: str - Success/error status
                - order: int - Order of execution
        """
        try:
            # Step 1: Get decomposition prompt
            prompt = self.get_decomposition_prompt(query)
            
            # Step 2: Get structured response from LLM using parse
            client = OpenAI()
            
            completion = client.beta.chat.completions.parse(
                model='gpt-4o',
                messages=[
                    {"role": "system", "content": "You are a query decomposition expert that outputs valid JSON matching the specified schema."},
                    {"role": "user", "content": prompt}
                ],
                response_format=DecompositionResponse
            )
            
            try:
                # Step 3: Parse JSON response
                response_data = json.loads(completion.choices[0].message.content)
                
                if 'sub_queries' not in response_data:
                    raise ValueError("Response missing 'sub_queries' field")
                
                # Step 4: Execute each sub-query and gather results
                results = []
                for sub_query in sorted(response_data['sub_queries'], key=lambda x: x['order']):
                    try:
                        api_response = self.api_manager.route_query(json.dumps(sub_query))
                        
                        # Add results and status to the sub-query
                        result = {
                            'api': sub_query['api'],
                            'params': sub_query['params'],
                            'order': sub_query['order'],
                            'results': api_response.get('results'),
                            'status': 'success' if 'error' not in api_response else 'error'
                        }
                        
                        if 'error' in api_response:
                            result['error'] = api_response['error']
                        else:
                            pass
                        
                        results.append(result)
                        
                    except Exception as e:
                        results.append({
                            'api': sub_query['api'],
                            'params': sub_query['params'],
                            'order': sub_query['order'],
                            'results': None,
                            'status': 'error',
                            'error': str(e)
                        })
                
                return results
                
            except Exception as e:
                return [{
                    'api': 'google_search',
                    'params': {'search_term': query},
                    'order': 1,
                    'results': None,
                    'status': 'error',
                    'error': f"Failed to process response: {str(e)}"
                }]
                
        except Exception as e:
            return [{
                'api': 'google_search',
                'params': {'search_term': query},
                'order': 1,
                'results': None,
                'status': 'error',
                'error': f"Query decomposition failed: {str(e)}"
            }]

    def get_decomposition_prompt(self, query: str) -> str:
        """
        INPUT:
            query: str - The user's query to be decomposed
            
        OUTPUT:
            str - Formatted prompt for decomposition model
        """
        try:
            prompt = f"""You are a query decomposition expert. Your task is to break down complex queries into simpler sub-queries that can be handled by specific APIs.

Available APIs and their parameters:
1. google_search:
   - search_term (str): The search query

2. get_stock_data:
   - symbol (str): Stock symbol (e.g., 'AAPL')
   - date (str, optional): Date in YYYY-MM-DD format

3. analyze_sentiment:
   - text (str): Text to analyze for sentiment

4. get_weather:
   - location (str): Location string (e.g., 'Palo Alto, CA')
   - date (str): Date in YYYY-MM-DD format

For the following query, break it down into sub-queries that can be handled by these APIs.

Original query: {query}

Respond with a JSON object that matches this schema:
{{
    "sub_queries": [
        {{
            "api": "google_search" | "get_stock_data" | "analyze_sentiment" | "get_weather",
            "params": {{
                "search_term"?: string,
                "symbol"?: string,
                "text"?: string,
                "location"?: string,
                "date"?: string
            }},
            "order": number
        }}
    ]
}}

Example response:
{{
    "sub_queries": [
        {{
            "api": "google_search",
            "params": {{
                "search_term": "specific search query"
            }},
            "order": 1
        }},
        {{
            "api": "analyze_sentiment",
            "params": {{
                "text": "text to analyze"
            }},
            "order": 2
        }}
    ]
}}

Ensure each sub-query:
1. Is specific and focused
2. Includes only the necessary parameters for that API
3. Has a logical execution order
4. Contributes to answering the original query

Your response:"""
            return prompt
        except Exception as e:
            return query

    def generate_prompt(self, query: str, decomposed_queries: List[Dict] = None) -> str:
        """
        INPUT:
            query: str - Original user query
            decomposed_queries: List[Dict] - List of decomposed query results
            
        OUTPUT:
            str - Enhanced prompt including original query and API results
        """
        try:
            # If no decomposed queries, return basic prompt
            if not decomposed_queries:
                return f"""Please answer the following query based on your knowledge:
                Query: {query}"""
                
                
            # Build context from decomposed queries
            context_parts = []
            
            online_LLM_knowledge = self.get_online_knowledge_using_LLM(query)
            
            # Sort queries by order to maintain logical flow
            for result in sorted(decomposed_queries, key=lambda x: x['order']):
                api_name = result['api']
                params = result['params']
                api_results = result.get('results')
                status = result['status']
                
                # Format the API call details
                param_str = ', '.join(f"{k}='{v}'" for k, v in params.items() if v)
                context_parts.append(f"\nAPI Call {result['order']}: {api_name}({param_str})")
                
                # Add the results or error message
                if status == 'success' and api_results:
                    if isinstance(api_results, (dict, list)):
                        context_parts.append(f"Results: {json.dumps(api_results, indent=2)}")
                    else:
                        context_parts.append(f"Results: {str(api_results)}")
                else:
                    error_msg = result.get('error', 'No results available')
                    context_parts.append(f"Status: {status}")
                    context_parts.append(f"Error: {error_msg}")
            
            # Construct the final prompt
            prompt = f"""Please answer the following query using the provided API results.
            
Original Query: {query}

Available Information:
{' '.join(context_parts)}

I am also providing you with some online knowledge that's hihgly reliable, and unless hihgly irrelevant this should have hihger precedent over other information, so give high priority to it to answer the queries.
Online Knowledge: {online_LLM_knowledge}

Based on the above information, please provide:
1. A comprehensive answer to the original query
2. Citations for each piece of information used (reference the specific API call number)
3. Confidence level in the answer (high/medium/low) with explanation

Your response should be well-structured and clearly indicate which API results were used to form the answer.

If any API calls failed or returned incomplete information, please acknowledge this in your response and explain how it affects the confidence in your answer.

Answer:"""
            
            return prompt
            
        except Exception as e:
            # Fallback to basic prompt
            return f"Please answer the following query: {query}"

    def iterative_refine(self, prompt: str, max_iterations: int = 3) -> List[str]:
        """
        INPUT:
            prompt: str - Input prompt for generation
            max_iterations: int - Maximum number of refinement iterations
            
        OUTPUT:
            List[str] - List of responses from each iteration, showing progression
        """
        responses = []  # Store responses from each iteration
        
        current_response = self.fuse(prompt)
        responses.append(current_response)
        try:
            for iteration in range(max_iterations):
                # If this is the last iteration, no need to verify
                if iteration == max_iterations - 1:
                    break
                # Pydantic model for verification response
                class VerificationResponse(BaseModel):
                    refined_query: str
                    continuation_decision: Literal["CONTINUE", "COMPLETE"]
                # Step 4: Verify response quality and identify gaps
                verification_prompt = f"""You are an AI assistant tasked with iteratively refining queries and search instructions to improve the quality of answers to complex questions. Your goal is to analyze previous responses, identify areas for improvement, and provide refined queries and specific search instructions.

Here is the original query:
<original_query>
{prompt}
</original_query>

The previous prompt used to generate an answer was:
<previous_prompt>
{self.decomposition_prompt_used}
</previous_prompt>

And the answer generated from that prompt was:
<previous_answer>
{current_response}
</previous_answer>

Analyze the previous answer carefully. Determine if it fully addresses the original query, if there are any inaccuracies, or if additional information is needed.

Based on your analysis, provide a refined query and specific search instructions. Your output should help guide the next iteration of information gathering and answer generation. Consider the following:

1. What aspects of the original query were not adequately addressed?
2. Are there any potential inaccuracies or inconsistencies in the previous answer?
3. What additional information would be helpful to provide a more comprehensive answer?
4. Are there any specific sources or types of information that should be prioritized in the next search?

Keep in mind that this is part of an iterative process. The maximum number of iterations is:
<max_iterations>
{max_iterations}
</max_iterations>

and current iteration is:
<current_iteration>
{iteration}
</current_iteration>

instruction for the refined query:
<refined_query>
[Write the refined query here. This should be a more specific or expanded version of the original query that addresses any gaps or issues identified in the previous answer. In this refined query, please incorporate any search instructions implicitly—include detailed guidance such as specific information to look for, potential sources to prioritize, and any additional context that may help the decomposer in the next iteration.]
</refined_query>

instruction for the continuation decision:
<continuation_decision>
[Write either "CONTINUE" if further refinement is needed and the maximum number of iterations has not been reached, or "COMPLETE" if the answer is satisfactory or the maximum number of iterations has been reached.]
</continuation_decision>

Provide your response in the following JSON format:
{{
    "continuation_decision": str
    "refined_query": str,
}}

Remember, your goal is to guide the iterative refinement process to produce the most accurate and comprehensive answer to the original query, so your produces query should be helpign towards that goal. Really try to deeply understand how the query impacts the search process by looking at the resopnse and the prompt, and the intial query, and use tha knowledge to craft your refined query."""
                
                completion = oaiclient.beta.chat.completions.parse(
                    model=self.iterative_refinement_model,
                    messages=[
                        {"role": "system", "content": verification_prompt},
                        {"role": "user", "content": f"Original query: {prompt}\nPrevious response: {current_response}"}
                    ],
                    response_format=VerificationResponse
                )
                
                verification_result = completion.choices[0].message.parsed
                
                
                try:
                    feedback = verification_result
                    
                    # If response is complete and accurate, we can stop
                    if feedback.continuation_decision == "COMPLETE":
                        break
                    
                    # Update prompt for next iteration with verification feedback
                    refined_query = feedback.refined_query
                    
                    current_response = self.fuse(refined_query)
                    responses.append(current_response)
                except json.JSONDecodeError:
                    # If we can't parse the verification result, just continue with the original prompt
                    prompt = f"Please improve this response further: {current_response}\nOriginal query: {prompt}"
                
            return responses
            
        except Exception as e:
            if not responses:
                # If we have no responses at all, generate at least one
                basic_response = self.fuse(prompt)
                responses.append(basic_response)
            return responses
    
    def fuse(self, prompt: str) -> str:
        """
        Queries multiple models with the same prompt and fuses the responses 
        by combining the best elements from each response.

        INPUT:
            prompt: str - Original prompt
            
        OUTPUT:
            str - Single fused response combining the best elements
        """
        try:
            # Define the models to use
            models = [
                "gpt-4o",
                "claude-3-5-haiku-latest",
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
            ]
            
            # Get responses from each model
            model_responses = []
            decomposed_queries = self.decompose_query(prompt)
            enhanced_prompt = self.generate_prompt(prompt, decomposed_queries)
            
            for model in models:
                try:
                    response = self.generate(enhanced_prompt, model)
                    if response:
                        model_responses.append({
                            "model": model,
                            "response": response
                        })
                except Exception as e:
                    pass
            
            if not model_responses:
                return "Error: Unable to get responses from models"
            
            # Create fusion prompt
            model_responses_text = "\n".join([f"=== {resp['model']} ===\n{resp['response']}\n" for resp in model_responses])
            
            fusion_prompt = f"""You are an expert at analyzing and combining multiple model responses into a single coherent answer. 
Below are responses from different models to the same query. Your task is to create a single comprehensive response that:
1. Combines the unique insights and information from each model
2. Resolves any contradictions between responses
3. Maintains a clear and coherent structure
4. Provides proper attribution for information sources
5. Indicates confidence levels for different parts of the response

Original Prompt:
{prompt}

Model Responses:
{model_responses_text}

Please provide a single comprehensive response that combines the best elements from all model responses while maintaining clarity and accuracy. 
If there are contradictions between model responses, explain how you resolved them.
If certain models provided unique insights, highlight those contributions.
Maintain the same format as the original responses (including citations and confidence levels if present in the original prompt). Keep it succinct though.
Make sure you include all the resources, and how the api infomration was used to answer the question.

Your fused response:"""
            
            # Get fused response using GPT-4o
            try:
                fused_response = self.generate(fusion_prompt, model=self.fusion_model)
                if not fused_response:
                    raise ValueError("Empty fusion response")
                    
                return fused_response
                
            except Exception as e:
                # Fallback to best individual response if fusion fails
                return model_responses[0]["response"]
                
        except Exception as e:
            return f"Error during fusion process: {str(e)}"

    def run_pipeline(self, query: str, iterations: int) -> str:
        """
        INPUT:
            query: str - User's natural language query
            iterations: int - Number of iterations of iterative refinement
        OUTPUT:
            str - Final fused response or error message
        """
        # pydantic base model
        class AnalysisResponse(BaseModel):
            explanation: str
            complexity_level: Literal["simple", "complex"]
            processing_method: Literal["single_call", "full_pipeline"]
            confidence: float
        
        # Create a prompt to analyze query complexity and determine processing approach
        analysis_prompt = f"""Analyze the following query and determine the best processing approach.

Query: {query}

Consider the following aspects:
1. Query Complexity:
   - Is this a simple, straightforward question?
   - Does it require multiple pieces of information?
   - Are there temporal or comparative elements?

2. API Requirements:
   - Can this be answered with a single API call?
   - Are multiple API calls needed?
   - What types of information are needed?

3. Processing Needs:
   - Is iterative refinement likely to improve the answer?
   - Would multiple model perspectives help?
   - Is information fusion needed?

Available Processing Methods:
1. single_LM_with_single_API_call: For simple queries requiring one API call
2. Full Pipeline (decompose + iterative refine + fuse): For complex queries needing multiple API calls or refinement

Provide your analysis in the following JSON format:
{{
    "explanation": str,
    "complexity_level": "simple" | "complex",
    "processing_method": "single_call" | "full_pipeline",
    "confidence": float  # 0.0 to 1.0
}}"""

        try:
            completion = oaiclient.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a query analysis expert that determines the best processing approach for user queries."},
                    {"role": "user", "content": analysis_prompt}
                ],
                response_format=AnalysisResponse
            )
            
            analysis = completion.choices[0].message.parsed
            
            # Process based on analysis
            if analysis.processing_method == "single_call":
                response = self.single_LM_with_single_API_call(query, 'gpt-4o')
                return response[0] if isinstance(response, list) else response
            else:
                initial_response = self.fuse(query)
                
                # Step 3: Iterative refinement
                recommended_iterations = min(iterations, 3)
                if recommended_iterations > 0:
                    refined_responses = self.iterative_refine(query, recommended_iterations)
                    # Return the last (most refined) response
                    return refined_responses[-1]
                
                return initial_response

        except Exception as e:
            # Fallback to single API call if something goes wrong
            response = self.single_LM_with_single_API_call(query, 'gpt-4o')
            return response[0] if isinstance(response, list) else response
