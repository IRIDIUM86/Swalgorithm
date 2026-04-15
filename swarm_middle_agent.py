import os
import sys
import time
import logging
import json
from colorama import Fore, Style, init
import google.genai as genai
import os
import json
import logging

# Initialize colorama
init(autoreset=True)

# =============================================================================
# Logging Configuration
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Custom Formatter for Logging that applies color based on log level
    and specific keywords.
    """
    LEVEL_COLORS = {
        logging.DEBUG: Fore.LIGHTYELLOW_EX,
        logging.INFO: Fore.WHITE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }
    KEYWORD_COLORS = {
        'HTTP Request': Fore.LIGHTYELLOW_EX,
    }

    def format(self, record):
        message = super().format(record)
        # Apply color based on specific keywords
        for keyword, color in self.KEYWORD_COLORS.items():
            if keyword in message:
                return color + message + Style.RESET_ALL
        # Otherwise, color based on log level
        color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        return color + message + Style.RESET_ALL

# Remove existing handlers to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create a console handler with the custom formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = ColoredFormatter('%(asctime)s %(levelname)s:%(message)s')
console_handler.setFormatter(console_formatter)

# Create a file handler for general logging
file_handler = logging.FileHandler("swarm_middle_agent.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
file_handler.setFormatter(file_formatter)

# Configure the root logger to use both handlers
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler],
)

# =============================================================================
# Swarm Client Initialization
# =============================================================================

def initialize_swarm_client():
    """
    Initializes the Gemini client.

    Returns:
        Client: An instance of the Gemini client.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logging.error("Gemini API key not found.")
            return None
        client = genai.Client(api_key=api_key)
        logging.info("Gemini client initialized successfully.")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client: {e}")
        sys.exit(1)

client = initialize_swarm_client()

# =============================================================================
# Constants & Agent Configuration
# =============================================================================

AGENTS_CONFIG_FILE = 'agents.json'

def load_agents_config():
    """
    Loads agent configurations from the 'agents.json' file.

    Returns:
        list: A list of agent configurations.
    """
    try:
        with open(AGENTS_CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded agents configuration from '{AGENTS_CONFIG_FILE}'.")
        return data.get('agents', [])
    except FileNotFoundError:
        logging.error(f"Agents configuration file '{AGENTS_CONFIG_FILE}' not found.")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing '{AGENTS_CONFIG_FILE}': {e}")
        return []

# =============================================================================
# Utility Functions
# =============================================================================

def print_divider(char="═", length=100, color=Fore.YELLOW):
    """
    Prints a divider line of specified character, length, and color.

    Args:
        char (str): The character to use for the divider.
        length (int): The length of the divider.
        color (str): The color code from colorama.
    """
    print(color + (char * length) + Style.RESET_ALL)

def print_header(title, color=Fore.YELLOW):
    """
    Prints a formatted header with a box around the title text.

    Args:
        title (str): The header title.
        color (str): The color code from colorama.
    """
    border = "═" * 58
    print(color + f"\n╔{border}╗")
    print(color + f"║{title.center(58)}║")
    print(color + f"╚{border}╝" + Style.RESET_ALL)

# =============================================================================
# Swarm Agents Initialization
# =============================================================================

def initialize_swarm_agents():
    """
    Initializes Swarm-based agents from configuration.

    Returns:
        list: A list of initialized Swarm Agent instances.
    """
    agents_data = load_agents_config()
    if not agents_data:
        logging.error("No agents found in the configuration. Please check 'agents.json'.")
        sys.exit(1)

    agents = []
    agent_data_dict = {}

    for agent_data in agents_data:
        name = agent_data.get('name', 'Unnamed Agent')
        system_purpose = agent_data.get('system_purpose', '')
        additional_attrs = {
            k: v for k, v in agent_data.items()
            if k not in ['name', 'system_purpose']
        }

        # Build the instructions from system purpose + other attributes
        full_instructions = system_purpose
        for attr_name, attr_value in additional_attrs.items():
            if isinstance(attr_value, dict):
                details = "\n".join(
                    f"{ak.replace('_',' ').title()}: {av}" for ak, av in attr_value.items()
                )
                full_instructions += f"\n\n{attr_name.replace('_',' ').title()}:\n{details}"
            else:
                full_instructions += f"\n\n{attr_name.replace('_',' ').title()}: {attr_value}"

        agent_dict = {
            'name': name,
            'instructions': full_instructions
        }
        agents.append(agent_dict)
        agent_data_dict[name] = agent_data

    # Inform agents about other agents
    for agent in agents:
        other_agents_info = ""
        for other_agent in agents:
            if other_agent['name'] != agent['name']:
                info = f"Name: {other_agent['name']}"
                o_data = agent_data_dict[other_agent['name']]
                sp = o_data.get('system_purpose', '')
                info += f"\nSystem Purpose: {sp}"

                more_attrs = {
                    k: v for k, v in o_data.items()
                    if k not in ['name', 'system_purpose']
                }
                for attr_name, attr_value in more_attrs.items():
                    if isinstance(attr_value, dict):
                        details = "\n".join(
                            f"{ak.replace('_',' ').title()}: {av}" for ak, av in attr_value.items()
                        )
                        info += f"\n{attr_name.replace('_',' ').title()}:\n{details}"
                    else:
                        info += f"\n{attr_name.replace('_',' ').title()}: {attr_value}"
                other_agents_info += f"\n\n{info}"

        agent['instructions'] += (
            f"\n\nYou are aware of the following other agents:\n{other_agents_info.strip()}"
        )

    logging.info(f"Initialized {len(agents)} agents.")
    return agents

# =============================================================================
# Swarm Reasoning Process
# =============================================================================

def run_swarm_reasoning(user_prompt):
    """
    Orchestrates a multi-agent 'Swarm' process to produce a refined or 
    consolidated answer to user_prompt, returning that final text.

    Args:
        user_prompt (str): The user's input prompt.

    Returns:
        str: The final blended response from the swarm.
    """
    agents = initialize_swarm_agents()
    num_agents = len(agents)
    opinions = {}
    verified_opinions = {}
    critiques = {}
    refined_opinions = {}

    print(Fore.YELLOW + "\nRunning Swarm-based reasoning...\n" + Style.RESET_ALL)

    # ------------------ Step 1: Discuss the Prompt ------------------
    print_header("Reasoning Step 1: Discussing the Prompt")
    for agent in agents:
        prompt = f"{agent.instructions}\n\nUser prompt: {user_prompt}"
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        agent_opinion = response.text.strip()
        opinions[agent.name] = agent_opinion
        color = get_agent_color(agent.name)
        print(color + f"{agent.name} response: {agent_opinion}" + Style.RESET_ALL)

    # ------------------ Step 2: Verify the Responses ------------------
    print_header("Reasoning Step 2: Verifying Responses")
    for agent in agents:
        verify_prompt = (
            f"{agent.instructions}\n\nPlease verify the accuracy of your previous response:\n\n{opinions[agent.name]}"
        )
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=verify_prompt
        )
        verified_opinion = response.text.strip()
        verified_opinions[agent.name] = verified_opinion
        color = get_agent_color(agent.name)
        print(color + f"{agent.name} verified response: {verified_opinion}" + Style.RESET_ALL)

    # ------------------ Step 3: Critique Each Other ------------------
    print_header("Reasoning Step 3: Critiquing Responses")
    for i, agent in enumerate(agents):
        other_agent = agents[(i + 1) % num_agents]
        critique_prompt = (
            f"{agent.instructions}\n\nPlease critique {other_agent.name}'s response "
            f"for depth and accuracy:\n\n{verified_opinions[other_agent.name]}"
        )
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=critique_prompt
        )
        critique_text = response.text.strip()
        critiques[agent.name] = critique_text
        color = get_agent_color(agent.name)
        print(color + f"{agent.name} critique on {other_agent.name}:\n{critique_text}\n" + Style.RESET_ALL)

    # ------------------ Step 4: Refine the Responses ------------------
    print_header("Reasoning Step 4: Refining Responses")
    for i, agent in enumerate(agents):
        other_agent = agents[(i + 1) % num_agents]
        refine_prompt = (
            f"{agent.instructions}\n\nPlease refine your response based on {other_agent.name}'s critique:\n\n"
            f"Your Original Response:\n{opinions[agent.name]}\n\n"
            f"{other_agent.name}'s Critique:\n{critiques[agent.name]}"
        )
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=refine_prompt
        )
        refined_text = response.text.strip()
        refined_opinions[agent.name] = refined_text
        color = get_agent_color(agent.name)
        print(color + f"{agent.name} refined response: {refined_text}" + Style.RESET_ALL)

    # ------------------ Step 5: Blend Refined Responses ------------------
    print_header("Reasoning Step 5: Blending Responses")
    agent_responses = [(agent.name, refined_opinions[agent.name]) for agent in agents]
    final_blended_response = blend_responses(agent_responses, user_prompt)
    print(Fore.GREEN + f"\nFinal Blended Response:\n{final_blended_response}" + Style.RESET_ALL)
    print(Fore.GREEN + "\nSwarm-based reasoning completed.\n" + Style.RESET_ALL)

    return final_blended_response

def blend_responses(agent_responses, user_prompt):
    """
    Combines multiple agent responses into a single, optimal response via a specialized 'Swarm Agent'.

    Args:
        agent_responses (list of tuples): (agent_name, response)
        user_prompt (str): The original prompt from the user.

    Returns:
        str: The blended optimal response text.
    """
    combined_prompt = (
        "Please combine the following responses into a single, optimal answer to the question.\n"
        f"Question: '{user_prompt}'\n"
        "Responses:\n"
        + "\n\n".join(
            f"Response from {agent_name}:\n{response}" for agent_name, response in agent_responses
        )
        + "\n\nProvide a concise and accurate combined response."
    )

    try:
        blender_instructions = "You are a collaborative AI assistant composed of multiple expert agents."
        combined_prompt_full = f"{blender_instructions}\n\n{combined_prompt}"

        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=combined_prompt_full
        )
        blended_reply = response.text.strip()

        logging.info("Blended response generated successfully.")
        return blended_reply
    except Exception as e:
        logging.error(f"Error in blend_responses: {e}")
        return "An error occurred while attempting to blend responses."

def get_agent_color(agent_name):
    """
    Retrieves the color associated with a given agent.

    Args:
        agent_name (str): The name of the agent.

    Returns:
        str: The color code from colorama.
    """
    agent_colors = {
        "Agent 47":     Fore.MAGENTA,
        "Agent 74":     Fore.CYAN,
        "Swarm Agent":  Fore.LIGHTGREEN_EX,
    }
    return agent_colors.get(agent_name, Fore.WHITE)

# =============================================================================
# Interface Functions
# =============================================================================

def swarm_middle_agent_interface(user_prompt):
    """
    Interface function to trigger multi-stage swarm reasoning with a single call
    from external code (e.g., reasoning.py).
    Returns the final swarm-blended text.

    Args:
        user_prompt (str): The user's input prompt.

    Returns:
        str: The final swarm response or None if an error occurred.
    """
    try:
        start_time = time.time()
        final_text = run_swarm_reasoning(user_prompt)
        end_time = time.time()
        logging.info(f"Swarm reasoning completed in {end_time - start_time:.2f} seconds.")
        return final_text
    except Exception as e:
        logging.error(f"Error in swarm_middle_agent_interface: {e}")
        return None

# =============================================================================
# Main (Optional Test)
# =============================================================================

if __name__ == "__main__":
    test_prompt = "What is love and how does it affect human behavior?"
    final = swarm_middle_agent_interface(test_prompt)
    if final:
        print(Fore.CYAN + f"\nSwarm final answer:\n{final}\n" + Style.RESET_ALL)
    else:
        print(Fore.CYAN + "No final swarm response captured." + Style.RESET_ALL)
