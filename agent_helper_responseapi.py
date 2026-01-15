#!/usr/bin/env python3
"""
Llama-Stack Responses API Agent Helper Module
Provides reusable functions for running agents using llama-stack Responses API.

The Responses API is a stateful system for multi-turn conversations with tool calling.
It maintains conversation state by chaining responses through previous_response_id.
"""

import json
import re
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

# Configure logging with absolute path
_log_file_path = Path(__file__).parent / 'llama_stack_api.log'
logger = logging.getLogger(__name__)

# Only configure if not already configured (avoid duplicate handlers)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(_log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"=== Logging initialized ===")
    logger.info(f"Log file: {_log_file_path}")

# OpenAI client for Responses API
from openai import OpenAI

# Suppress SSL warnings
import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


def extract_json_from_response(response_text):
    """
    Extract JSON from a response that might contain markdown formatting
    
    Args:
        response_text (str): Response text that might contain JSON in markdown code blocks
    
    Returns:
        dict: Parsed JSON data or None if extraction fails
    """
    if not response_text or not isinstance(response_text, str):
        return None
    
    # Clean up the response text
    response_text = response_text.strip()
    
    try:
        # First try to parse the entire response as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json ... ```
        r'```json\s*(.*?)```',      # ```json ... ``` (no newline)
        r'```\s*\n(.*?)\n```',      # ``` ... ```
        r'```\s*(.*?)```',          # ``` ... ``` (no newline)
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                cleaned = match.strip()
                if cleaned:
                    return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON-like content in the response
    # Look for content that starts with { and ends with }
    # Use a more sophisticated pattern that handles nested braces
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(response_text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # Found a complete JSON object
                json_str = response_text[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Continue searching for another JSON object
                    start_idx = -1
                    brace_count = 0
                    continue
    
    # Try to find JSON array - if found and contains objects, extract first object
    bracket_count = 0
    start_idx = -1
    for i, char in enumerate(response_text):
        if char == '[':
            if start_idx == -1:
                start_idx = i
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0 and start_idx != -1:
                json_str = response_text[start_idx:i+1]
                try:
                    parsed = json.loads(json_str)
                    # If it's an array with objects, return the first object
                    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                        return parsed[0]
                    return parsed
                except json.JSONDecodeError:
                    start_idx = -1
                    bracket_count = 0
                    continue
    
    # Last resort: try to find any JSON-like structure
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    return None


def _get_mcp_server_url(tool_group: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Map tool_group to MCP server URL
    
    Args:
        tool_group (str): Tool group identifier (e.g., "mcp::openshift", "mcp::atlassian")
        config (dict): Configuration dictionary
    
    Returns:
        str: MCP server URL or None if not found
    """
    # Check for explicit MCP server URLs in config
    mcp_servers = config.get('mcp_servers', {})
    
    # Map tool_group to server name
    server_mapping = {
        'mcp::openshift': 'openshift',
        'mcp::atlassian': 'atlassian',
        'mcp::slack': 'slack',
    }
    
    server_name = server_mapping.get(tool_group)
    if server_name and server_name in mcp_servers:
        return mcp_servers[server_name]
    
    # Fallback: try environment variables
    env_mapping = {
        'mcp::openshift': os.getenv('MCP_OPENSHIFT_SERVER_URL', 'http://ocp-mcp-server:8000/sse'),
        'mcp::atlassian': os.getenv('MCP_ATLASSIAN_SERVER_URL', 'http://atlassian-mcp-server:8080/sse'),
        'mcp::slack': os.getenv('MCP_SLACK_SERVER_URL'),
    }
    
    return env_mapping.get(tool_group)


def _get_tool_group_for_responses_api(tool_group: str, config: Dict[str, Any]) -> Optional[list]:
    """
    Convert tool_group to Responses API tools format
    
    Args:
        tool_group (str): Tool group identifier (e.g., "mcp::openshift", "mcp::atlassian")
        config (dict): Configuration dictionary
    
    Returns:
        list: List of tool configurations for Responses API, or None if not found
    """
    # Responses API uses MCP tools with server_url
    # Format: [{"type": "mcp", "server_label": "...", "server_url": "..."}]
    
    # Get MCP server URL
    server_url = _get_mcp_server_url(tool_group, config)
    if not server_url:
        return None
    
    # Map tool_group to server label
    server_mapping = {
        'mcp::openshift': 'openshift',
        'mcp::atlassian': 'atlassian',
        'mcp::slack': 'slack',
    }
    
    server_label = server_mapping.get(tool_group, 'mcp')
    
    # Return tool configuration for Responses API
    return [{
        "type": "mcp",
        "server_label": server_label,
        "server_url": server_url
    }]


def run_agent(system_prompt: str, user_prompt: str, tool_group: str, 
              config: Dict[str, Any], max_infer_iters: int = 20,
              node_name: str = None, system_prompt_id: str = None, 
              user_prompt_id: str = None) -> Dict[str, Any]:
    """
    Run an agent using llama-stack Responses API
    
    Args:
        system_prompt (str): System instructions for the agent (used as instructions parameter)
        user_prompt (str): User prompt/query (used as input)
        tool_group (str): Tool group to use (e.g., "mcp::atlassian", "mcp::openshift")
        config (dict): Configuration dictionary with:
            - openai_endpoint or base_url: OpenAI-compatible API endpoint
            - model: Model name
            - api_key: API key (default: "fake")
            - temperature: Temperature setting (default: 0.7)
            - max_tokens: Max tokens (default: 4096, not used in Responses API)
        max_infer_iters (int): Maximum inference iterations (default: 20)
        node_name (str, optional): Node name for logging purposes
        system_prompt_id (str, optional): System prompt ID for logging purposes
        user_prompt_id (str, optional): User prompt ID for logging purposes
    
    Returns:
        dict: Response with success status, final answer, and raw response
    """
    console = Console()
    
    try:
        # Get OpenAI-compatible endpoint from config
        # llama-stack OpenAI endpoint is at /v1/openai/v1
        openai_endpoint = config.get('openai_endpoint') or config.get('base_url')
        if not openai_endpoint:
            raise ValueError("openai_endpoint or base_url must be provided in config")
        
        # If endpoint doesn't include /v1/openai/v1, add it
        if '/v1/openai/v1' not in openai_endpoint:
            # Remove any existing /v1 or /v1/ at the end
            openai_endpoint = openai_endpoint.rstrip('/')
            if openai_endpoint.endswith('/v1'):
                openai_endpoint = openai_endpoint[:-3]
            # Add /v1/openai/v1
            openai_endpoint = f"{openai_endpoint}/v1/openai/v1"
        
        # Ensure URL doesn't end with trailing slash
        openai_endpoint = openai_endpoint.rstrip('/')
        
        # Initialize OpenAI client for Responses API
        client = OpenAI(
            base_url=openai_endpoint,
            api_key=config.get('api_key', 'fake'),
        )
        
        # Get tools configuration for Responses API
        tools = _get_tool_group_for_responses_api(tool_group, config)
        
        # Log API request details
        logger.info(f"=== Llama-Stack Responses API Request ===")
        if node_name:
            logger.info(f"Node Name: {node_name}")
        if system_prompt_id:
            logger.info(f"System Prompt ID: {system_prompt_id}")
        if user_prompt_id:
            logger.info(f"User Prompt ID: {user_prompt_id}")
        logger.info(f"API Endpoint: {openai_endpoint}")
        logger.info(f"Model: {config.get('model', 'N/A')}")
        logger.info(f"Tool Group: {tool_group}")
        logger.info(f"Raw Query (System Prompt/Instructions):\n{system_prompt}")
        logger.info(f"Raw Query (User Prompt/Input):\n{user_prompt}")
        
        # Create response using Responses API
        # The Responses API uses:
        # - input: user message(s) - can be string or list of messages
        # - instructions: system prompt
        # - tools: list of tool configurations
        #
        # Note: max_infer_iters is NOT a supported parameter for OpenAI Responses API.
        # We keep it in the function signature for compatibility with other helpers,
        # but we do not pass it to the API client.
        response = client.responses.create(
            model=config.get('model', 'llama-4-scout-17b-16e-w4a16'),
            input=user_prompt,  # Can be string or list of message dicts
            instructions=system_prompt,
            tools=tools,
            temperature=config.get('temperature', 0.7),
            stream=False,
        )
        
        # Log raw API response
        logger.info(f"=== Llama-Stack Responses API Response ===")
        logger.info(f"Response ID: {response.id}")
        logger.info(f"Response Status: {response.status}")
        logger.info(f"Response Output Type: {type(response.output).__name__}")
        
        # Extract final response from output
        # Responses API provides output_text property for convenience
        final_response = ""
        tool_executed = False
        tool_message_count = 0
        raw_messages = []
        
        # Try to get output_text directly (convenience property)
        if hasattr(response, 'output_text') and response.output_text:
            final_response = response.output_text
        else:
            # Fallback: Process output items manually
            for output_item in response.output:
                if hasattr(output_item, 'type'):
                    if output_item.type == 'message':
                        # Extract text content from message
                        if hasattr(output_item, 'content') and output_item.content:
                            # Content is a list of content items
                            for content_item in output_item.content:
                                if hasattr(content_item, 'text'):
                                    if not final_response:
                                        final_response = content_item.text
                                    raw_messages.append({
                                        "role": "assistant",
                                        "content": content_item.text
                                    })
                    elif output_item.type == 'tool_call':
                        # Tool was called
                        tool_executed = True
                        tool_message_count += 1
                        raw_messages.append({
                            "role": "tool",
                            "content": str(output_item)
                        })
        
        # Count tool calls from output items if not already counted
        if not tool_executed and response.output:
            for output_item in response.output:
                if hasattr(output_item, 'type') and output_item.type == 'tool_call':
                    tool_executed = True
                    tool_message_count += 1
        
        # If still no response, try to serialize the response object
        if not final_response:
            try:
                # Try to get JSON representation
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict() if hasattr(response, 'dict') else {}
                final_response = json.dumps(response_dict, indent=2, default=str)
            except Exception:
                final_response = str(response)
        
        # Log final processed response
        logger.info(f"Final Answer:\n{final_response}")
        logger.info(f"Tool Executed: {tool_executed}, Tool Message Count: {tool_message_count}")
        logger.info(f"=== Llama-Stack Responses API Response End ===")
        
        return {
            "success": True,
            "final_answer": final_response,
            "raw_response": json.dumps({
                "response_id": response.id,
                "status": response.status,
                "output": [item.model_dump() if hasattr(item, 'model_dump') else str(item) for item in response.output]
            }, indent=2, default=str),
            "session_id": response.id,  # Use response ID as session identifier
            "messages": raw_messages,
            "tool_executed": tool_executed,
            "tool_message_count": tool_message_count,
        }
        
    except Exception as e:
        error_msg = f"Responses API Agent failed ({type(e).__name__}): {str(e)}"
        console.print(f"[red]❌ {error_msg}[/red]")
        # Print full exception for debugging
        import traceback
        console.print(f"[dim]Full traceback:[/dim]")
        for line in traceback.format_exception(type(e), e, e.__traceback__)[-10:]:
            console.print(f"[dim]{line.rstrip()}[/dim]")
        logger.error(f"Error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": error_msg,
            "final_answer": None,
            "raw_response": None,
            "session_id": None,
        }
