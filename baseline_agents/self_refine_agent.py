import contextlib
import io
import logging
import traceback
import multiprocessing
import re
import sys
import numpy as np

from typing import Optional, Tuple, Dict, Any, List, Literal
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.utils.interactive_env import is_interactive_env
from langgraph.graph import StateGraph, START, END

import openai
from falsification_agent.llm.custom_model import CustomChatModel
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)


CODING_AGENT_SYSTEM_PROMPT = """You are an expert statistician. You are tasked to validate rigorously if a scientific hypothesis H is true by implementing a rigorous statistical test. 

You should write code to implement a falsification test for the given hypothesis. 
The test should be relevant to the main hypothesis and aims to falsify it. 
The test should use the available data described below, and use data processing, extraction, and perform statistical analysis to produce a p-value measuring the falsification of the main hypothesis. 
The test should be extremely rigorous. The p-value should be theoretically grounded.
The code should be clear, concise, and efficient. Do progress bar when necessary. It will have a time limit, so please be efficient. For example, if possible, you can set the number of permutations to be small (e.g. <1000).
The code should be self-contained, and do not need additional modifications from user.

Create a code from the user request. Ensure any code you provide can be executed with all required imports and variables defined. 
Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block. 
Invoke the CodeOutputSpec tool to structure the output correctly. 
NEVER PRODUCE ANY PLACEHOLDER IN ANY FUNCTION. PLACEHOLDER IS WORSE THAN FAILURE TO PRODUCE CODE.
PLACEHOLDER including coming up with placeholder genes, names, ids, functions, p-value, or any other placeholder.
The output should be a single p-value. If there are multiple p-values produced by the test, you should aggregate them in a meaningful and rigorous way.
When printing p-values, please use scientific notations (e.g. 3.50e-03) instead of the raw number.
-------------------------------------------------------

Here is the user requested falsification test specification:"""

FEEDBACK_AGENT_SYSTEM_PROMPT = """You are an evaluateGPT. Given a test specification, an implementation of the test, and the results from executing the implemented test, your task is to evaluate if the test implementation and output is valid and provide detailed feedback on any identified issues and suggestions for improving the test.

To evaluate the validity of the test, you should
1. Check if the test implementation strictly follows the test specification
2. Make sure the output shows a valid p-value without any errors. The p-value should have a reasonable value: it cannot be smaller than or equal to 0 or larger than 1.
3. Double-check that all data inputs used in the experiment are accessed from the provided data sources, and there are no fake/made-up data entries.
4. Examine carefully through the experiment implementation; make sure it uses rigorous statistical test and there are no bugs or logical issues in the code.
5. Carefully examine any other potential problems of the experiment, such as unhandled edge-cases, invalid sample sizes, or other subtle issues that might lead to a misleading result.

If you find any problems with the test implementation or experiment output, please provide a detailed feedback on how to fix and refine the test. If the test is valid, please output the final p-value formatted in scientific notation."""

def get_llm(model = 'claude-3-5-sonnet-20240620', temperature=0.7, **kwargs):
    source = "custom"
    if model[:7] == 'claude-':
        source = 'Anthropic'
    elif model[:4] == 'gpt-':
        source = 'OpenAI'
    elif model.startswith('llama'):
        source = "Llama"
    # if source not in ['OpenAI', 'Anthropic']:
    #     raise ValueError('Invalid source')
    if source == 'OpenAI':
        return ChatOpenAI(model = model, temperature = temperature, **kwargs)
    elif source == 'Anthropic':
        return ChatAnthropic(model = model, 
                            temperature = temperature,
                            max_tokens = 4096,
                            **kwargs)
    elif source == 'Llama':
        llm = CustomChatModel(model = model, model_type=source, temperature = temperature)
        llm.client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY").chat.completions
        return llm
    else:
        # listen to a different port
        llm = CustomChatModel(model = model, model_type=source, temperature = temperature)
        llm.client = openai.Client(base_url="http://127.0.0.1:40000/v1", api_key="EMPTY").chat.completions
        return llm


class FeedbackOutputSpec(BaseModel):
    """Output specification for the experiment evaluation & feedback"""
    
    is_valid: str = Field(
        description="The validity of the experiment implementation and output. Use 'Yes' if the experiment is valid, or 'No' if any issues are detected."
    )
    p_value: Optional[str] = Field(description="The p-value from the experiment, formatted in scientific notation. Include this field only if the experiment is valid.")
    feedback: str = Field(
        description="Detailed feedback to address identified issues and suggestions for improving the experiment."
    )

class CodeOutputSpec(BaseModel):
    """Code output"""
    
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


class SelfRefineState(TypedDict):
    iteration: int
    messages: List[Tuple[str, str]]
    code_impl: Optional[CodeOutputSpec]
    code_out: Optional[str]
    feedback_valid: bool
    p_value: Optional[float]
    feedback_text: Optional[str]
    done: bool


class CodingAgent:
    """
    - Takes test_specification & data as input.
    - Produces code using CODING_AGENT_SYSTEM_PROMPT + user specification.
    - Executes the code and captures the output.
    """

    def __init__(self, llm="claude-3-5-sonnet-20241022", time_limit: int = 3):
        """
        Args:
            llm: name of the language model
            time_limit: Time limit (in minutes) to run code.
        """
        self.llm = get_llm(llm)
        self.time_limit = time_limit

        # We will build a ChatPromptTemplate for code generation
        # Then parse the result with CodeOutputSpec.
        self.code_generation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    CODING_AGENT_SYSTEM_PROMPT,
                ),
                ("placeholder", "{messages}"),
            ]
        )

        self.structured_llm = self.llm.with_structured_output(CodeOutputSpec)
        
        self._exec_globals = {}
        self._exec_globals.update(__builtins__)
    
    def _set_globals(self, table_dict=None):
        self._exec_globals = {}
        self._exec_globals.update(__builtins__)
        
        if table_dict:
            self._exec_globals.update(table_dict)
    
    def generate_code(self, messages) -> CodeOutputSpec:
        """Generate code from the LLM given a test specification."""
        # Invoke the pipeline
        code_gen_chain = self.code_generation_prompt | self.structured_llm
        result: CodeOutputSpec = code_gen_chain.invoke({"messages": messages})

        return result

    def execute_code(self, code_implementation: CodeOutputSpec) -> str:
        """
        Safely executes the provided code in a subprocess (with a time limit).
        Returns stdout/stderr as a string.
        """
        # Combine imports + code
        full_code = code_implementation.imports + "\n\n" + code_implementation.code
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().setLevel(logging.INFO)

        # Worker function to run code in a separate process
        def run_code(queue):
            output_capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
                    logging.getLogger().handlers[0].stream = output_capture
                    exec_globals = self._exec_globals
                    exec(full_code, exec_globals)
            except Exception:
                error_message = traceback.format_exc()
                queue.put(error_message)
            else:
                queue.put(output_capture.getvalue())

        # Use multiprocessing to enforce time limit
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=run_code, args=(queue,))
        proc.start()
        proc.join(timeout=self.time_limit * 60)

        if proc.is_alive():
            # Timed out
            proc.terminate()
            proc.join()
            return "TimeoutError: Code execution took too long."

        # Retrieve captured output
        if not queue.empty():
            code_output = queue.get()
        else:
            code_output = "No output was captured."
        return code_output
    
    def run(
        self,
        messages: Any
    ) -> Tuple[CodeOutputSpec, str]:
        """
        High-level method:
        1) Generate code from LLM
        2) Execute the code
        3) Return code + output
        """
        code_impl = self.generate_code(messages)
        print("---------Code Implementation-----------")
        print(code_impl)
        code_out = self.execute_code(code_impl)
        print("----------Code Output-----------")
        print(code_out)
        return code_impl, code_out

class FeedbackAgent:
    """
    - Takes in the test specification, data, code implementation, and execution result.
    - Evaluates using FEEDBACK_AGENT_SYSTEM_PROMPT.
    - Produces a FeedbackOutputSpec indicating whether the test is valid, a final p-value if valid, and suggestions.
    """

    def __init__(self, llm="claude-3-5-sonnet-20241022"):
        self.llm = get_llm(llm)

        # Build a ChatPromptTemplate for feedback
        self.feedback_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FEEDBACK_AGENT_SYSTEM_PROMPT),
                ("user", "{evaluation_input}"),
            ]
        )
        self.structured_llm = self.llm.with_structured_output(FeedbackOutputSpec)

    def run(
        self,
        test_specification: str,
        data: str,
        code_implementation: CodeOutputSpec,
        code_output: str,
    ) -> FeedbackOutputSpec:
        """
        1) Concatenate specification + code + output into a single message
        2) Ask LLM for feedback with FEEDBACK_AGENT_SYSTEM_PROMPT
        3) Parse into FeedbackOutputSpec
        """
        spec_text = (
            f"Main Hypothesis: {test_specification}\n"
        )
        
        code_text = (
            f"--- Code Implementation ---\n"
            f"Prefix:\n{code_implementation.prefix}\n\n"
            f"Imports:\n{code_implementation.imports}\n\n"
            f"Code:\n{code_implementation.code}\n\n"
        )

        output_text = f"--- Execution Result ---\n{code_output}\n\n"

        # Summarize data if needed (or directly pass it)
        # For brevity, we just pass a note that data is available:
        data_text = f"Data description:\n{data}\n\n"

        evaluation_input = spec_text + code_text + output_text + data_text

        # Now run the LLM
        final_chain = self.feedback_prompt | self.structured_llm
        feedback_result: FeedbackOutputSpec = final_chain.invoke(
            {"evaluation_input": evaluation_input}
        )
        
        print("---------Feedback----------")
        print(feedback_result)
        return feedback_result

class SelfRefineAgent:
    """
    - Orchestrates iterative refinement:
      1) Generate code with CodingAgent
      2) Evaluate with FeedbackAgent
      3) If invalid, refine or retry until success or max attempts
      4) Once valid, parse p-value < 0.05 => {"output": True} else {"output": False}.
    """

    def __init__(
        self,
        llm: str = "claude-3-5-sonnet-20241022",
        max_iterations: int = 10,
        p_value_threshold: float = 0.05,
    ):
        self.coding_agent = CodingAgent(llm)
        self.feedback_agent = FeedbackAgent(llm)
        self.max_iterations = max_iterations
        self.p_value_threshold = p_value_threshold
        
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Create and compile the LangGraph for the iterative refinement.
        """
        def generate_and_run_code(state: SelfRefineState):
            """
            Node that calls coding_agent.run(...) to produce code & execution results.
            """
            code_impl, code_out = self.coding_agent.run(
                state["messages"]
            )
            state["code_impl"] = code_impl
            state["code_out"] = code_out
            state["messages"] += [
                (
                    "assistant",
                    f"{code_impl.prefix} \n Imports: {code_impl.imports} \n Code: {code_impl.code}",
                )
            ]
            return state

        def evaluate_feedback(state: SelfRefineState):
            """
            Node that calls feedback_agent.run(...) to see if code is valid.
            """
            feedback = self.feedback_agent.run(
                self.test_specification,
                self.data,
                state["code_impl"],
                state["code_out"]
            )
            # Suppose feedback has structure:
            # {
            #   "is_valid": "Yes" or "No",
            #   "p_value": "...",
            #   "feedback": "..."
            # }
            # We store these in state
            state["feedback_valid"] = (feedback.is_valid.lower() == "yes")
            if feedback.is_valid.lower() == "yes":
                # attempt to parse p-value
                try:
                    pval = float(feedback.p_value) if feedback.p_value else None
                except:
                    pval = None
                state["p_value"] = pval
            else:
                state["p_value"] = None
            state["feedback_text"] = feedback.feedback
            state["messages"] += [
                ("user", feedback.feedback)
            ]
            return state

        def decide_if_done(state: SelfRefineState) -> Literal["check_p_value", "maybe_retry"]:
            """
            Decide if the feedback was valid. If valid, move to next step. If not, maybe keep iterating.
            """
            if state["feedback_valid"]:
                state["done"] = True
                return "end"
            else:
                return "check_max_iteration"

        def check_max_iteration(state: SelfRefineState) -> Literal["generate_and_run_code", "end"]:

            state["iteration"] += 1
            if state["iteration"] > self.max_iterations:
                state["done"] = True
            return state
        
        def maybe_retry(state: SelfRefineState) -> Literal["generate_and_run_code", "end"]:
            if state["done"] == True:
                return "end"
            return "generate_and_run_code"

        graph_builder = StateGraph(SelfRefineState)

        graph_builder.add_node("generate_and_run_code", generate_and_run_code)
        graph_builder.add_node("evaluate_feedback", evaluate_feedback)
        graph_builder.add_node("check_max_iteration", check_max_iteration)

        graph_builder.add_edge(START, "generate_and_run_code")
        graph_builder.add_edge("generate_and_run_code", "evaluate_feedback")
        graph_builder.add_conditional_edges(
            "evaluate_feedback",
            decide_if_done,
            {
                "end": END,
                "check_max_iteration": "check_max_iteration",
            }
        )
        graph_builder.add_conditional_edges(
            "check_max_iteration",
            maybe_retry,
            {
                "generate_and_run_code": "generate_and_run_code",
                "end": END,
            }
        )

        return graph_builder.compile()
    
    def generate(
        self,
        query: str,
        data_loader: Any,
    ) -> Dict[str, Any]:
        """
        Returns:
          Dict with keys:
            - 'output': bool (True if p-value < threshold, False otherwise)
            - 'feedback': feedback text from the final iteration
            - 'p_value': (optional) p-value from the final iteration
        """
        self.test_specification = query
        self.data = data_loader.data_desc
        self.coding_agent._set_globals(data_loader.table_dict)
        messages = [
            ("user", "Here is the hypothesis to falsify:" + self.test_specification + "\n\n" + "And here are the available data relevant to the hypothesis:\n" + self.data + "\n\nEach of these dataframes have already been loaded into the global namespace. You may access each dataframe **directly as variables**. Make sure to use the **EXACT** dataframe names as shown above.")
        ]
        
        state_dict = SelfRefineState(
            iteration=1,
            messages=messages,
            code_impl=None,
            code_out=None,
            feedback_valid=False,
            p_value=None,
            feedback_text=None,
            done=False
        )
        # run the graph
        result = self.graph.invoke(state_dict)

        # interpret final results
        if result["p_value"] is not None and result["p_value"] < self.p_value_threshold:
            return {
                "output": True,
                "feedback": result["feedback_text"],
                "p_value": result["p_value"],
            }
        elif result["p_value"] is not None:
            return {
                "output": False,
                "feedback": result["feedback_text"],
                "p_value": result["p_value"],
            }
        else:
            # no valid p-value or max iterations exhausted
            return {
                "output": False,
                "feedback": "No valid p-value found or max iteration limit reached."
            }