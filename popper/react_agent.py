from popper.react_utils import create_agent
from popper.prompt_utils import get_react_coding_agent_system_prompt
from popper.llm.custom_model import CustomChatModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import os
import json
import langchain
import openai
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
import logging
import uuid
import traceback
import io
import contextlib
import sys
import re

# uncomment the following line to enable debug mode
# langchain.debug = True

class LiveLogger:
    """Custom stdout handler that logs in real-time while also printing output."""
    def __init__(self, log):
        self.original_stdout = sys.stdout  # Store original stdout
        self.log = log  # Log dictionary
        self.current_buffer = []  # Store intermediate logs

    def clean_message(self, message):
        """Remove ANSI escape codes and filter out unnecessary logs."""
        # Remove ANSI escape codes (color formatting)
        message = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', message)

        # Filter out specific unwanted messages
        unwanted_logs = [
            "> Entering new AgentExecutor chain...",
            "> Finished chain."
        ]
        if any(unwanted in message for unwanted in unwanted_logs):
            return None  # Skip logging this message

        return message.strip() if message.strip() else None

    def write(self, message):
        cleaned_message = self.clean_message(message)
        if cleaned_message:
            self.original_stdout.write(cleaned_message + "\n")  # Print to console
            self.original_stdout.flush()  # Ensure immediate output
            
            # Append each new log update separately in Markdown format
            self.current_buffer.append(cleaned_message)
            self.log['executor'].append(f"```\n{cleaned_message}\n```")  # Markdown formatting

    def flush(self):
        self.original_stdout.flush()


def get_prompt_data(
        prompt_config: str = None
):
    if prompt_config is None and os.environ.get("PROMPT_CONFIG") is None:
        raise ValueError("PROMPT_CONFIG not set and prompt_config not provided")
    else:
        prompt_config = prompt_config or os.environ.get("PROMPT_CONFIG")

    with open(prompt_config, "r") as file:
        return json.load(file)


class ReactAgent():
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        model_config: str = None,
        api_config: str = None,
        max_iterations: int = 25,
        prompt_revision: bool = False,
        port=None,
        api_key="EMPTY",
    ):
        self.prompt_revision = prompt_revision
        self.api = "custom"
        if model_name[:7] == 'claude-':
            self.api = 'anthropic'
        elif model_name[:4] == 'gpt-':
            self.api = 'openai'
        else:
            self.api = 'local'
        
        # logger.add(log_file, format="{time} {level} {message}", level="INFO")
        self.stdout_handler = StdOutCallbackHandler()

        # set max iterations
        self.max_iterations = max_iterations

        # load model config
        self.model_name = model_name

        # get the model
        self.llm = self.get_model(
            api=self.api,
            model=self.model_name,
            port=port,
            api_key=api_key
        )

        # create agent
        self.agent = create_agent(
            llm=self.llm,
            handlers=[self.stdout_handler],
            max_iterations=self.max_iterations
        )

    def get_model(
            self,
            api,
            model,
            port=None,
            api_key=None,
            **kwargs
    ):
        llm = None
        if (api == "anthropic"):
            llm = ChatAnthropic(
                model=model,
                api_key=os.environ["ANTHROPIC_API_KEY"],
                **kwargs
            )
        elif (api == "openai"):
            llm = ChatOpenAI(
                model=model,
                api_key=os.environ["OPENAI_API_KEY"],
                **kwargs
            )
        # elif (api == 'llama'):
        #     llm = CustomChatModel(
        #         model=model,
        #         model_type='custom',
        #         **kwargs
        #     )
        #     llm.client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY").chat.completions
        else:
            # Llama or other locally-served models
            assert port is not None, "Port must be specified for local models"
            llm = CustomChatModel(
                model=model,
                model_type='custom',
                **kwargs
            )
            api_key = "EMPTY" if api_key is None else api_key
            llm.client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key=api_key).chat.completions
        return llm
        
    def generate(self, data_loader, test_spec, domain, log=None):
        try:
            self.agent.tools[0]._set_globals(data_loader.table_dict)
            dataset_desc = data_loader.data_desc
            
            # Use LiveLogger only if a log is provided
            logger = LiveLogger(log) if log is not None else sys.stdout

            # Redirect stdout to capture real-time logs
            sys.stdout = logger
            try:
                output = self.agent.invoke(input={
                    "system_prompt": get_react_coding_agent_system_prompt(domain=domain, prompt_revision=self.prompt_revision),
                    "input": f"""Falsification Test: {test_spec}
    Datasets: {dataset_desc}
    Thought:"""
                })
            finally:
                sys.stdout = logger.original_stdout  # Restore stdout

            return output['output']

        except Exception as e:
            error_message = f"Execution Stopped due to: {e}\n{traceback.format_exc()}"
            print(error_message)
            if log is not None:
                log['executor'].append(f"```\n{error_message}\n```")  # Markdown format
            return None
    '''
    def generate(self, data_loader, test_spec, domain, log = None):
        try:
            self.agent.tools[0]._set_globals(data_loader.table_dict)
            dataset_desc = data_loader.data_desc
            output = self.agent.invoke(input={
                "system_prompt": get_react_coding_agent_system_prompt(domain=domain, prompt_revision=self.prompt_revision),
                "input": f"""Falsification Test: {test_spec}
Datasets: {dataset_desc}
Thought:"""
            })
            return output['output']
        except Exception as e:
            print("Execution Stopped due to : ", e)
            print(traceback.format_exc())
            return None    
    '''
