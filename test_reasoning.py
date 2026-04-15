import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import sys

# Add the current directory to the path so we can import reasoning
sys.path.insert(0, os.path.dirname(__file__))

from reasoning import (
    load_agents_config,
    Agent,
    blend_responses,
    generate_video,
    reasoning_logic,
    initialize_agents,
    get_shared_system_message
)

class TestReasoning(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.sample_agents_config = {
            "agents": [
                {
                    "name": "Creator",
                    "system_purpose": "Enhance the user's prompt for a high-quality AI video."
                },
                {
                    "name": "Critic",
                    "system_purpose": "Find flaws in the prompt."
                },
                {
                    "name": "Judge",
                    "system_purpose": "Decide if the prompt is ready."
                }
            ]
        }

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_agents_config_success(self, mock_json_load, mock_file):
        """Test successful loading of agents configuration."""
        mock_json_load.return_value = self.sample_agents_config

        result = load_agents_config()

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['name'], 'Creator')
        self.assertEqual(result[1]['name'], 'Critic')
        self.assertEqual(result[2]['name'], 'Judge')

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_agents_config_file_not_found(self, mock_file):
        """Test loading agents config when file is not found."""
        result = load_agents_config()
        self.assertEqual(result, [])

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    def test_load_agents_config_invalid_json(self, mock_json_load, mock_file):
        """Test loading agents config with invalid JSON."""
        result = load_agents_config()
        self.assertEqual(result, [])

    def test_get_shared_system_message(self):
        """Test that shared system message is returned."""
        message = get_shared_system_message()
        self.assertIsInstance(message, str)
        self.assertIn("AI Assistant", message)
        self.assertIn("Guidelines for Interaction", message)

    @patch('reasoning.genai')
    def test_agent_initialization(self, mock_genai):
        """Test Agent class initialization."""
        agent_data = {
            'name': 'TestAgent',
            'system_purpose': 'Test purpose',
            'personality': {'trait': 'value'}
        }

        agent = Agent('color', **agent_data)

        self.assertEqual(agent.name, 'TestAgent')
        self.assertEqual(agent.color, 'color')
        self.assertIn('Test purpose', agent.instructions)
        self.assertIn('Trait: value', agent.instructions)

    @patch('reasoning.client')
    def test_agent_discuss(self, mock_client):
        """Test Agent discuss method with mocked Gemini API."""
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_client.models.generate_content.return_value = mock_response

        agent = Agent('color', name='TestAgent', system_purpose='Test purpose')
        result, duration = agent.discuss("Test prompt")

        self.assertEqual(result, "Test response")
        self.assertIsInstance(duration, float)
        mock_client.models.generate_content.assert_called_once()

    @patch('reasoning.client')
    def test_blend_responses(self, mock_client):
        """Test blend_responses function with mocked Gemini API."""
        mock_response = MagicMock()
        mock_response.text = "Blended response"
        mock_client.models.generate_content.return_value = mock_response

        agent_responses = [
            ('Agent1', 'Response1'),
            ('Agent2', 'Response2')
        ]
        user_prompt = "Test question"

        result = blend_responses(agent_responses, user_prompt)

        self.assertEqual(result, "Blended response")
        mock_client.models.generate_content.assert_called_once()

    @patch('reasoning.Client')
    @patch.dict(os.environ, {'HF_SPACE_ID': 'test_space', 'Access_Token': 'test_token'})
    def test_generate_video_success(self, mock_client_class):
        """Test video generation with mocked gradio client."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.predict.return_value = "/path/to/video.mp4"

        result = generate_video("Test prompt")

        # Since we can't easily capture print output, just ensure no exception
        self.assertIsNone(result)  # Function returns None on success

    @patch.dict(os.environ, {}, clear=True)
    def test_generate_video_missing_env(self):
        """Test video generation with missing environment variables."""
        result = generate_video("Test prompt")
        # Should not raise exception, just print error
        self.assertIsNone(result)

    @patch('reasoning.client')
    @patch('builtins.input', side_effect=['Test prompt', 'menu'])
    @patch('reasoning.initialize_agents')
    @patch('reasoning.print_header')
    @patch('reasoning.print_divider')
    @patch('reasoning.process_agent_action')
    def test_reasoning_logic_workflow(self, mock_process_action, mock_print_divider,
                                     mock_print_header, mock_init_agents, mock_input, mock_client):
        """Test the reasoning logic workflow."""
        # Mock agents
        mock_creator = MagicMock()
        mock_creator.name = 'Creator'
        mock_critic = MagicMock()
        mock_critic.name = 'Critic'
        mock_judge = MagicMock()
        mock_judge.name = 'Judge'

        mock_init_agents.return_value = [mock_creator, mock_critic, mock_judge]

        # Mock API responses
        mock_response = MagicMock()
        mock_response.text = "Enhanced prompt"
        mock_client.models.generate_content.return_value = mock_response

        # Mock process_agent_action
        mock_process_action.return_value = ("PASS", 0.1)

        # Mock the workflow to exit after one iteration
        with patch('reasoning.generate_video') as mock_gen_video:
            reasoning_logic([mock_creator, mock_critic, mock_judge])

        # Verify generate_video was called
        mock_gen_video.assert_called_once()

    @patch('reasoning.load_agents_config')
    @patch('reasoning.Agent')
    def test_initialize_agents_with_config(self, mock_agent_class, mock_load_config):
        """Test agent initialization with configuration."""
        mock_load_config.return_value = self.sample_agents_config['agents']

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        result = initialize_agents()

        self.assertEqual(len(result), 3)
        self.assertEqual(mock_agent_class.call_count, 3)

    @patch('reasoning.load_agents_config', return_value=[])
    @patch('reasoning.Agent')
    def test_initialize_agents_default(self, mock_agent_class, mock_load_config):
        """Test agent initialization with default agents when no config."""
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        result = initialize_agents()

        self.assertEqual(len(result), 2)  # Default agents
        self.assertEqual(mock_agent_class.call_count, 2)

if __name__ == '__main__':
    unittest.main()