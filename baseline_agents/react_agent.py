from baseline_agents.react_utils import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_together import Together
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
import langchain
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from baseline_agents.utils.dv_log import DVLogger
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
        model_config: str = None,
        api_config: str = None,
        model_name: str = "claude-3-5-sonnet",
        log_file: str = "base_react_agent.log",
        max_iterations: int = 25,
        simple_prompt: bool = False
    ):
        self.logfile = log_file
        self.logger = DVLogger(f"{model_name}_{uuid.uuid4()}", log_file)
        # logger.add(log_file, format="{time} {level} {message}", level="INFO")
        self.file_handler = FileCallbackHandler(self.logfile)
        self.stdout_handler = StdOutCallbackHandler()

        # set max iterations
        self.max_iterations = max_iterations

        # check if model config is provided
        if model_config is None and os.environ.get("MODEL_CONFIG") is None:
            raise ValueError("MODEL_CONFIG not set and model_config not provided")
        else:
            # override environment variable config path if model_config is provided
            model_config = model_config or os.environ.get("MODEL_CONFIG")

        # do a similar check for api_config
        if api_config is None and os.environ.get("API_CONFIG") is None:
            raise ValueError("API_CONFIG not set and api_config not provided")
        else:
            api_config = api_config or os.environ.get("API_CONFIG")


        # load model config
        self.model_name = model_name
        with open(model_config, "r") as file:
            self.model_config = json.load(file)

        # load api config
        with open(api_config, "r") as file:
            self.api_config = json.load(file)

        try:
            # get full model name and type
            self.full_model_name = self.model_config['models'][self.model_name]['model_name']
            self.model_type = self.model_config['models'][self.model_name]['model_type']
        except KeyError:
            raise ValueError(f"Model {model_name} not found in model config")

        try:
            # get api key using model type
            self.api_key = self.api_config[self.model_type]
        except KeyError:
            raise ValueError(f"API key not found for {self.model_type}")

        # get the model
        self.llm = self.get_model(
            api=self.model_type,
            model=self.full_model_name,
            api_key=self.api_key
        )

        # create agent
        self.agent = create_agent(
            llm=self.llm,
            handlers=[self.file_handler, self.stdout_handler],
            max_iterations=self.max_iterations,
            simple_template=simple_prompt
        )

    def get_model(
            self,
            api,
            api_key,
            model,
            **kwargs
    ):
        llm = None
        if (api == "together"):
            llm = Together(
                model=model,
                together_api_key=api_key,
                **kwargs
            )
        elif (api == "anthropic"):
            llm = ChatAnthropic(
                model=model,
                api_key=api_key,
                **kwargs
            )
        elif (api == "openai"):
            llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                **kwargs
            )
        elif (api == "google"):
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                **kwargs
            )
        else:
            raise ValueError(f"Invalid API: {api}")
        return llm

    def generate(self, data_loader, query):
        try:
            table_dict = data_loader.table_dict
            dataset_desc = data_loader.data_desc
            self.agent.tools[0]._set_globals(table_dict)
            output = self.agent.invoke(input={
                "system_prompt": "You are a scientific agent who can plan and execute python code multiple times to answer a query based on one or more datasets. All datasets have already been loaded into the global namespace as pandas dataframes.",
                "input": f"""Question: Is the following hypothesis true or false? {query}
Datasets: {dataset_desc}"""
            })
            self.logger.log_json(output)
            return output
        except Exception as e:
            print("Execution Stopped due to : ", e)
            print(traceback.format_exc())
            self.logger.logger.error(f"Execution Stopped due to : {e}")
        self.logger.close()