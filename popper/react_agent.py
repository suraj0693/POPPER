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


# uncomment the following line to enable debug mode
# langchain.debug = True

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
    ):
        self.prompt_revision = prompt_revision
        self.api = "custom"
        if model_name[:7] == 'claude-':
            self.api = 'anthropic'
        elif model_name[:4] == 'gpt-':
            self.api = 'openai'
        elif model_name.startswith('llama'):
            self.api = 'llama'
        
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
        elif (api == 'llama'):
            llm = CustomChatModel(
                model=model,
                model_type='custom',
                **kwargs
            )
            llm.client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY").chat.completions
        else:
            # Mixtral or other custom model
            llm = CustomChatModel(
                model=model,
                model_type='custom',
                **kwargs
            )
            llm.client = openai.Client(base_url="http://127.0.0.1:40000/v1", api_key="EMPTY").chat.completions
        return llm

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