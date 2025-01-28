# Set up the base template
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_react_agent
from langchain.prompts import StringPromptTemplate
from langchain.tools import BaseTool
from langchain.chains.llm import LLMChain
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.schema import AgentAction, AgentFinish
from pydantic import Field, PrivateAttr
from typing import List, Union, Dict
import contextlib
import io
import logging
import re

logging.basicConfig(level=logging.INFO)

template = """{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Falsification Test: description of a hypothesis falsification test that you need to implement
Datasets: the names and descriptions of datasets relevant to the input falsification test
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final output from the falsification test (i.e., whether you are able to reject the null hypothesis with statistical significance). Make sure to also include the p-value of the statistical test written in scientific notations.

IMPORTANT: Please make sure the Final Answer includes the p-value of the falsification test regardless if you are able to reject the null hypothesis. **Only return the Final Answer if you have obtained a non-zero p-value**. When printing p-values, please use scientific notations instead of the raw number.

IMPORTANT: Please avoid p-hacking! Be fair and rigorous.

Note: all datasets have already been loaded into the global namespace as Pandas dataframes. You may access the data by referring to the EXACT dataframe names as provided in the "Datasets:" section.

--------------------------------------------
Example
Falsification Test:
{{
    "Falsification Test name": "Body Length Evolution and Speciation Rate Relationship Test",
    "Falsification Test description": "Testing for a significant positive relationship between maximum body length evolution rate and spatial variation in speciation rates.",
    "Falsification Test Null hypothesis": "There is no statistically significant positive relationship between the rate of maximum body length evolution and spatial variation in speciation rates.",
    "Falsification Test Alternate hypothesis": "There is a statistically significant positive relationship between the rate of maximum body length evolution and spatial variation in speciation rates."
}}
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
Action Input: from scipy.stats import linregress\n\n# Perform linear regression to test for a statistically significant relationship\nresult = linregress(df['BAMM_speciation'], df['BAMM_NetDiv'])\ncoefficient = result.slope\np_value = result.pvalue\ncoefficient, "{{:.2e}}".format(p_value)
Observation: (0.5175306498596297, 3.50e-03)
Thought: The linear regression analysis provides a coefficient of approximately 0.518, indicating a positive relationship, and the p-value is 3.50e-03, which is statistically significant at the 0.05 level. Based on this, I can reject the null hypothesis in the falsification test.
Final Answer: Falsification test passes. The null hypothesis is rejected with a p-value of 3.50e-03.
--------------------------------------------

Remember, your output should always **exactly** follow the aforementioned format:
Falsification Test: description of a hypothesis falsification test that you need to implement
Datasets: the names and descriptions of datasets relevant to the input falsification test
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final output from the falsification test (i.e., whether you are able to reject the null hypothesis with statistical significance). Make sure to also include the p-value of the statistical test written in scientific notations.

**IMPORTANT**
You should ALWAYS report a p-value EXACTLY AS IT IS. If a p-value is 4.2e-01, report 4.2e-01, DO NOT REPORT 4.2e-02!
BE CAREFUL WHEN READING THE P-VALUE RESULTS, MISREPORTING A P-VALUE IS WORSE THAN HAVING NO P-VALUE AT ALL.
When reading p-values, make sure the sample sizes and the statistical test is valid.
Please make sure to always return ONE valid p-value. If there are multiple p-values produced by the test, aggregate them in a meaningful and rigorous way.
** Always make sure the returned p-value matches your conclusion for the falsification test. For example, if you reject H0 but finds out that H1 is also incorrect (e.g., the suggested shape or relationship is wrong), you SHOULD NOT return a p-value < 0.05.
If you think it's impossible to find a valid p-value for the falsification test, return a p-value of 1.00e+00.
DO NOT perform p-hacking.

Begin!

{input} {agent_scratchpad}"""


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
        prompt = self.template.format(**kwargs)
        # print([prompt])
        return prompt

# CustomOutputParser to parse the output of the LLM and execute actions
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            print(f"Warning: could not parse LLM output: `{llm_output}`, finishing chain...")
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
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
        
        if code.endswith("Observation"):
            code = code[:-len("Observation")].rstrip()
        
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
    max_iterations = 50,
    early_stopping_method: str = "force",
):
    output_parser = CustomOutputParser()
    python_tool = CustomPythonAstREPLTool(callbacks=handlers)
    tools = [python_tool]
    tool_names = [tool.name for tool in tools]

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["system_prompt", "input", "intermediate_steps", "tool_names", "tools", "agent_scratchpad"]
    )
    # llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=handlers)

    # agent = LLMSingleActionAgent(
    #     llm_chain=llm_chain,
    #     output_parser=output_parser,
    #     stop=["\nObservation:"],
    #     allowed_tools=tool_names
    # )
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=output_parser,
        stop_sequence=["\nObservation:"],
    )

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        callbacks=handlers,
        early_stopping_method=early_stopping_method
    )