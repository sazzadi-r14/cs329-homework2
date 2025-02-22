from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel
import json
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = OpenAI()

def prepare_dataset(max_rows=None, debug_mode=False):
    """
    Load the CS329A HW2 evaluation dataset.
    
    Args:
        max_rows (int): If provided, only load first max_rows rows.
        debug_mode (bool): If True, only load first 10 rows. Defaults to False.
        
    Returns:
        Dataset: The loaded dataset, potentially truncated if in debug mode
    """
    dataset = load_dataset("ScalingIntelligence/cs329A_hw2_evaluation")
    
    if max_rows:
        dataset['train'] = dataset['train'].select(range(max_rows))
    elif debug_mode:
        dataset['train'] = dataset['train'].select(range(10))
        
    return dataset

def evaluate_qa(queries, responses, answers):
    """
    Evaluate QA performance by checking if answer words are contained in responses.
    
    Args:
        queries (list): List of question strings
        responses (list): List of model response strings
        answers (list): List of ground truth answer strings
        
    Returns:
        tuple: (accuracy percentage, dict with successful and unsuccessful cases)
    """
    if len(queries) != len(responses) or len(queries) != len(answers):
        raise ValueError("All input lists must have the same length")
        
    results = {
        "successful": [],
        "unsuccessful": []
    }
    
    correct = 0
    total = len(queries)
    
    from tqdm import tqdm
    for i, (query, response, answer) in enumerate(tqdm(zip(queries, responses, answers), total=len(queries))):
        # Convert response and answer to lowercase
        if (type(response) == list):
            response = response[-1]
        response_lower = response.lower()
        answer_lower = answer.lower()
        
        # Split answer into words and remove punctuation
        answer_words = set(word.strip('.,!?()[]{}":;') for word in answer_lower.split())
        
        
        ## Changing this part with AI because it's often time doing wrong evaluation for because of formatting.
        
        #pydantic basemodel of the response
        class EvaluationResponse(BaseModel):
            reasoning: str
            correct: bool
        
        prompt = f"""
        You are an expert evaluator. You are given a question, a response, and the ground truth answer.
        You need to determine if the response contains all the words in the answer.
        
        Question: {query}
        Response: {response}
        Answer: {answer}
        
        Keep in mind that response might be bigger and more verbose than the answer. You will say correct or incorrect based on the fact if the response contains the answer at any shape or form at all, or if it says something totally wrong.
        Return your response in the following format json:
        {{
            "reasoning": str,
            "correct": bool
        }}
        """
        
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {query}\nResponse: {response}\nAnswer: {answer}"}
            ],
            response_format=EvaluationResponse
        )
        
        result = completion.choices[0].message.parsed
        is_correct = result.correct
        reasoning = result.reasoning
        
        if is_correct:
            correct += 1
            results["successful"].append({
                "query": query,
                "response": response,
                "answer": answer
            })
        else:
            results["unsuccessful"].append({
                "query": query,
                "response": response,
                "answer": answer
            })
    
    accuracy = (correct / total) * 100
    
    return accuracy, results

# Example usage:
if __name__ == "__main__":
    # Example test cases
    test_queries = [
        "Who received the IEEE Frank Rosenblatt Award in 2010?",
        "What's the name of the women's liberal arts college in Cambridge, Massachusetts?"
    ]
    test_responses = [
        "The IEEE Frank Rosenblatt Award in 2010 was awarded to Michio Sugeno.",
        "Harvard University is in Cambridge."
    ]
    test_answers = [
        "Michio Sugeno",
        "Radcliffe College"
    ]
    
    accuracy, results = evaluate_qa(test_queries, test_responses, test_answers)
    
    # Print detailed results
    print("\nDetailed Results:")
    print("Successful cases:")
    for case in results["successful"]:
        print(f"Q: {case['query']}")
        print(f"R: {case['response']}")
        print(f"A: {case['answer']}\n")
        
    print("Unsuccessful cases:")
    for case in results["unsuccessful"]:
        print(f"Q: {case['query']}")
        print(f"R: {case['response']}")
        print(f"A: {case['answer']}\n")