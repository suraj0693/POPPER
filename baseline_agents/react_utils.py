# Set up the base template
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.tools import BaseTool
from langchain.chains.llm import LLMChain
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from typing import List, Union, Dict
from langchain.schema import AgentAction, AgentFinish
from pydantic import Field, PrivateAttr
import contextlib
import io
import logging
import re

logging.basicConfig(level=logging.INFO)

template = """{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Question: an input hypothesis that you must decide if it is True or False
Datasets: the names and descriptions of datasets relevant to the input hypothesis
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
WORKFLOW SUMMARY: this is the workflow that I used to find the final answer
Final Answer: True/False. Please output True if the input hypothesis is valid (e.g., you are able to reject the null hypothesis with statistical significance) and False if the input hypothesis is invalid (e.g., if you fail to reject the null hypothesis).

Please make sure the Final Answer is either True or False. Also generate a summary of the full workflow starting from data loading that led to the final answer as "WORKFLOW SUMMARY:"

IMPORTANT: all datasets have already been loaded into the global namespace as Pandas dataframes. You may access the data by referring to the EXACT dataframe names as provided in the "Datasets:" section.


Example
Question: Is the following hypothesis True or False? There is a statistically significant positive relationship between the rate of maximum body length evolution and spatial variation in speciation rates.
Datasets:
{{
    "name": "df_body-size-evolution-in-south-american-freshwater-fishes",
    "description": "Data on body size evolution in South American freshwater fishes, including speciation and extinction rates",
    "columns": {{
        "raw": [
            {{
                "name": "HYBAS_ID",
                "description": "Unique identifier for each hydrological basin"
            }},
            {{
                "name": "long",
                "description": "Longitude of the basin location"
            }},
            {{
                "name": "lat",
                "description": "Latitude of the basin location"
            }},
            {{
                "name": "BAMM_speciation",
                "description": "Rate of speciation as calculated by the BAMM method"
            }},
            {{
                "name": "BAMM_extinction",
                "description": "Rate of extinction as calculated by the BAMM method"
            }},
            {{
                "name": "BAMM_NetDiv",
                "description": "Net diversification rate, calculated as speciation minus extinction"
            }},
            {{
                "name": "aet",
                "description": "Mean annual evapotranspiration for each basin"
            }},
            {{
                "name": "Elevation",
                "description": "Average elevation of the basin"
            }},
            {{
                "name": "sgr",
                "description": "Species growth rate in each basin"
            }},
            {{
                "name": "soil_div",
                "description": "Soil diversity index for each basin"
            }},
            {{
                "name": "area",
                "description": "Total area of the basin in square kilometers"
            }},
            {{
                "name": "diversity",
                "description": "Diversity index for the species in each basin"
            }}
        ]
    }}
}}
Thought: First, I need to load the dataset from the global namespace in Python and inspect the data to identify the relevant columns for this hypothesis test.
Action: python_repl_ast
Action Input: import pandas as pd\n\ndf = df_body-size-evolution-in-south-american-freshwater-fishes\ndf.head()
Observation: 
     HYBAS_ID       long       lat  BAMM_speciation  BAMM_extinction  BAMM_NetDiv  ...   aet    Elevation  sgr  soil_div     area  diversity
0  6050000010 -76.477422  7.742693         0.137392         0.026807     0.110585  ...  1387   330.150088  166  0.482402  72363.7         68
1  6050000740 -74.628725  9.803586         0.117235         0.025796     0.091438  ...  1082    69.475294   23  0.457436  17944.3         35
2  6050068100 -75.295995  8.448815         0.119381         0.023826     0.095555  ...  1312   143.032178   74  0.378793  17105.5         44
3  6050068110 -74.608408  8.922863         0.132477         0.027777     0.104700  ...  1445    14.724138    3  0.468328    610.1         48
4  6050070260 -75.591588  5.770093         0.120127         0.022940     0.097187  ...  1371  1378.729945  421  0.158870  61901.9         81
[5 rows x 21 columns]
Thought: Now that the dataset is loaded and I can see the columns, I need to perform a statistical test to assess the significance of the relationship between 'BAMM_speciation' and 'BAMM_NetDiv'.
Action: python_repl_ast
Action Input: from scipy.stats import linregress\n\n# Perform linear regression to test for a statistically significant relationship\nresult = linregress(df['BAMM_speciation'], df['BAMM_NetDiv'])\ncoefficient = result.slope\np_value = result.pvalue\ncoefficient, p_value
Observation: (0.5175306498596297, 0.0035)
Thought: The linear regression analysis provides a coefficient of approximately 0.518, indicating a positive relationship, and the p-value is 0.0035, which is statistically significant at the 0.05 level. Based on this, I can conclude that the hypothesis is true.
WORKFLOW SUMMARY:
1. Data Loading: Loaded the dataset from the global namespace using Python.
2. Data Inspection: Displayed the first few rows of the dataset to confirm relevant columns.
3. Statistical Analysis: Performed a linear regression analysis between 'BAMM_speciation' (predictor) and 'BAMM_NetDiv' (response). The analysis yielded a positive coefficient and a p-value of 0.0035, indicating statistical significance.
Final Answer: True


Begin!

{input}
{agent_scratchpad}"""


template_v2 = """{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Question: an input hypothesis that you must decide if it is True or False
Datasets: the names and descriptions of datasets relevant to the input hypothesis
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
WORKFLOW SUMMARY: this is the workflow that I used to find the final answer
Final Answer: True/False. Please output True if you believe the input hypothesis is correct and False if the input hypothesis is not based on your analysis.

Please make sure the Final Answer is either True or False. Also generate a summary of the full workflow starting from data loading that led to the final answer as "WORKFLOW SUMMARY:"

IMPORTANT: all datasets have already been loaded into the global namespace as Pandas dataframes. You may access the data by referring to the EXACT dataframe names as provided in the "Datasets:" section.


Example
Question: Is the following hypothesis True or False? There is a statistically significant positive relationship between the rate of maximum body length evolution and spatial variation in speciation rates.
Datasets:
{{
    "name": "df_body-size-evolution-in-south-american-freshwater-fishes",
    "description": "Data on body size evolution in South American freshwater fishes, including speciation and extinction rates",
    "columns": {{
        "raw": [
            {{
                "name": "HYBAS_ID",
                "description": "Unique identifier for each hydrological basin"
            }},
            {{
                "name": "long",
                "description": "Longitude of the basin location"
            }},
            {{
                "name": "lat",
                "description": "Latitude of the basin location"
            }},
            {{
                "name": "BAMM_speciation",
                "description": "Rate of speciation as calculated by the BAMM method"
            }},
            {{
                "name": "BAMM_extinction",
                "description": "Rate of extinction as calculated by the BAMM method"
            }},
            {{
                "name": "BAMM_NetDiv",
                "description": "Net diversification rate, calculated as speciation minus extinction"
            }},
            {{
                "name": "aet",
                "description": "Mean annual evapotranspiration for each basin"
            }},
            {{
                "name": "Elevation",
                "description": "Average elevation of the basin"
            }},
            {{
                "name": "sgr",
                "description": "Species growth rate in each basin"
            }},
            {{
                "name": "soil_div",
                "description": "Soil diversity index for each basin"
            }},
            {{
                "name": "area",
                "description": "Total area of the basin in square kilometers"
            }},
            {{
                "name": "diversity",
                "description": "Diversity index for the species in each basin"
            }}
        ]
    }}
}}
Thought: First, I need to load the dataset from the global namespace in Python and inspect the data to identify the relevant columns for this hypothesis test.
Action: python_repl_ast
Action Input: import pandas as pd\n\ndf = df_body-size-evolution-in-south-american-freshwater-fishes\ndf.head()
Observation: 
     HYBAS_ID       long       lat  BAMM_speciation  BAMM_extinction  BAMM_NetDiv  ...   aet    Elevation  sgr  soil_div     area  diversity
0  6050000010 -76.477422  7.742693         0.137392         0.026807     0.110585  ...  1387   330.150088  166  0.482402  72363.7         68
1  6050000740 -74.628725  9.803586         0.117235         0.025796     0.091438  ...  1082    69.475294   23  0.457436  17944.3         35
2  6050068100 -75.295995  8.448815         0.119381         0.023826     0.095555  ...  1312   143.032178   74  0.378793  17105.5         44
3  6050068110 -74.608408  8.922863         0.132477         0.027777     0.104700  ...  1445    14.724138    3  0.468328    610.1         48
4  6050070260 -75.591588  5.770093         0.120127         0.022940     0.097187  ...  1371  1378.729945  421  0.158870  61901.9         81
[5 rows x 21 columns]
Thought: Now that the dataset is loaded and I can see the columns, I need to perform a statistical test to assess the significance of the relationship between 'BAMM_speciation' and 'BAMM_NetDiv'.
Action: python_repl_ast
Action Input: from scipy.stats import linregress\n\n# Perform linear regression to test for a statistically significant relationship\nresult = linregress(df['BAMM_speciation'], df['BAMM_NetDiv'])\ncoefficient = result.slope\np_value = result.pvalue\ncoefficient, p_value
Observation: (0.5175306498596297, 0.0035)
Thought: The linear regression analysis provides a coefficient of approximately 0.518, indicating a positive relationship, and the p-value is 0.0035, which is statistically significant at the 0.05 level. Based on this, I can conclude that the hypothesis is true.
WORKFLOW SUMMARY:
1. Data Loading: Loaded the dataset from the global namespace using Python.
2. Data Inspection: Displayed the first few rows of the dataset to confirm relevant columns.
3. Statistical Analysis: Performed a linear regression analysis between 'BAMM_speciation' (predictor) and 'BAMM_NetDiv' (response). The analysis yielded a positive coefficient and a p-value of 0.0035, indicating statistical significance.
Final Answer: True


Begin!

{input}
{agent_scratchpad}"""


def load_data_to_react_globals(data_loader):
    for name, df in data_loader.table_dict.items():
            globals()[name] = df


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# CustomOutputParser to parse the output of the LLM and execute actions
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            output = llm_output.split("Final Answer:")[-1].split()[0].strip().lower()
            if output not in ["true", "false", "yes", "no", "y", "n"]:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            return AgentFinish(
                return_values={"output": output in ["true", "yes", 'y']},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


class CustomPythonAstREPLTool(PythonAstREPLTool):
    _exec_globals:Dict = PrivateAttr()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize a persistent global namespace for code execution
        self._exec_globals = {}
        self._exec_globals.update(__builtins__)
    
    def _set_globals(self, table_dict=None):
        self._exec_globals = {}
        self._exec_globals.update(__builtins__)
        
        if table_dict:
            self._exec_globals.update(table_dict)
        
    def _run(self, query: str, run_manager=None):
        code_match = re.search(r"```(.*?)```", query, re.DOTALL)
        if code_match:
            # Extract code within backticks
            code = code_match.group(1)
        else:
            code = query
        code = code.strip()
        if code.startswith("python"):
            code = code[len("python"):].lstrip()
        
        code_lines = code.strip().split('\n')
        code = '\n'.join(code_lines[:-1])   # avoid printing the last line twice
        last_line = code_lines[-1]
        
        output_capture = io.StringIO()
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            logging.getLogger().handlers[0].stream = output_capture
            try:
                exec(code, self._exec_globals)
                try:
                    result = eval(last_line, self._exec_globals)
                    if result is not None:
                        print(result, file=output_capture)
                except:
                    pass
            except Exception as e:
                return str(e)
        
        # Retrieve the output and return it
        output = output_capture.getvalue()
        return output if output else "Execution completed without output."


def create_agent(
    llm,
    handlers,
    max_iterations = None,
    early_stopping_method: str = "force",
    simple_template = False
):
    output_parser = CustomOutputParser()
    python_tool = CustomPythonAstREPLTool(callbacks=handlers)
    tools = [python_tool]
    tool_names = [tool.name for tool in tools]

    if simple_template:
        use_template = template_v2
    else:
        use_template = template

    prompt = CustomPromptTemplate(
        template=use_template,
        tools=tools,
        input_variables=["system_prompt", "input", "intermediate_steps"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=handlers)

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        callbacks=handlers,
        early_stopping_method=early_stopping_method
    )