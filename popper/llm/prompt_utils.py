import json

def bind_tools_to_system_prompt(system_prompt, tools, tool_choice = None):
    if not tool_choice or tool_choice == 'none':
        return f'''You are an intelligent agent capable of calling tools to complete user-assigned tasks.
Here are the instructions specified by the user:
"""{system_prompt}"""

In addition, you have access to the following tools:
{json.dumps(tools, indent=4)}

You may output any intermediate thoughts or reasonings before delivering your final response. 
Your final response must either be at least one tool call or a response message to the user.

To make one or more tool calls, wrap your final response in the following JSON format:
{{
    "type": "tool_calls",
    "content": [
        {{
            "name": "name of the function to call",
            "id": "an unique id for this tool call",
            "arguments": {{
                "argument1": value1,
                "argument2": value2,
                ...
            }}
        }},
        ...
    ]
}}

To send a direct response message to the user, wrap your final response in the following JSON format:
{{
    "type": "text_message",
    "content": "content of the message according to the user instructions"
}}

You must choose either to send tool calls or a direct response message. Be sure to format the final response properly according to the given JSON specs.

DO NOT put anything after the final response JSON object.
'''

    # tool choice!
    system_prompt = f'''You are an intelligent agent capable of calling tools to complete user-assigned tasks.
Here are the instructions specified by the user:
"""{system_prompt}"""

In addition, you have access to the following tools:
{json.dumps(tools, indent=4)}

You may output any intermediate thoughts or reasonings before delivering your final response. 
Your final response MUST BE one or more tool calls.

To make one or more tool calls, wrap your final response in the following JSON format:
{{
    "type": "tool_calls",
    "content": [
        {{
            "name": "name of the function to call",
            "id": "an unique id for this tool call",
            "arguments": {{
                "argument1": value1,
                "argument2": value2,
                ...
            }}
        }},
        ...
    ]
}}


You MUST wrap your response as a tool call formatted in the above JSON schema.

DO NOT put anything after the tool call.
'''
    return system_prompt