#!/usr/bin/env python3
"""
OpenAI-Compatible Agent Helper Module
Provides reusable functions for running LangChain 1.0 agents with an OpenAI-compatible LLM endpoint and MCP tools.

This module is used when AGENT_ENGINE=openai-compatible.
"""

import json
import re
import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

# Configure logging with absolute path
_log_file_path = Path(__file__).parent / "llama_stack_api.log"
logger = logging.getLogger(__name__)

# Only configure if not already configured (avoid duplicate handlers)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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

    logger.info("=== Logging initialized ===")
    logger.info(f"Log file: {_log_file_path}")

# LangChain 1.0 imports
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# MCP Adapters for client-side tool execution
from langchain_mcp_adapters.client import MultiServerMCPClient

# Suppress SSL warnings
import warnings

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


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
        r"```json\s*\n(.*?)\n```",  # ```json ... ```
        r"```json\s*(.*?)```",  # ```json ... ``` (no newline)
        r"```\s*\n(.*?)\n```",  # ``` ... ```
        r"```\s*(.*?)```",  # ``` ... ``` (no newline)
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
        if char == "{":
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # Found a complete JSON object
                json_str = response_text[start_idx : i + 1]
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
        if char == "[":
            if start_idx == -1:
                start_idx = i
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
            if bracket_count == 0 and start_idx != -1:
                json_str = response_text[start_idx : i + 1]
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
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text, re.DOTALL)
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
    mcp_servers = config.get("mcp_servers", {})

    # Map tool_group to server name
    server_mapping = {
        "mcp::openshift": "openshift",
        "mcp::atlassian": "atlassian",
        "mcp::slack": "slack",
    }

    server_name = server_mapping.get(tool_group)
    if server_name and server_name in mcp_servers:
        return mcp_servers[server_name]

    # Fallback: try environment variables
    env_mapping = {
        "mcp::openshift": os.getenv("MCP_OPENSHIFT_SERVER_URL", "http://ocp-mcp-server:8000/sse"),
        "mcp::atlassian": os.getenv("MCP_ATLASSIAN_SERVER_URL", "http://atlassian-mcp-server:8080/sse"),
        "mcp::slack": os.getenv("MCP_SLACK_SERVER_URL"),
    }

    return env_mapping.get(tool_group)


async def _run_agent_async(
    system_prompt: str,
    user_prompt: str,
    tool_group: str,
    config: Dict[str, Any],
    node_name: str = None,
    system_prompt_id: str = None,
    user_prompt_id: str = None,
) -> Dict[str, Any]:
    """
    Async implementation of run_agent using LangChain 1.0 create_agent
    """
    console = Console()

    try:
        # Get OpenAI-compatible endpoint from config
        # llama-stack OpenAI endpoint is at /v1/openai/v1
        # ChatOpenAI will add /chat/completions automatically, so we need /v1/openai/v1
        openai_endpoint = config.get("openai_endpoint") or config.get("base_url")
        if not openai_endpoint:
            raise ValueError("openai_endpoint or base_url must be provided in config")

        # If endpoint doesn't include /v1/openai/v1, add it
        if "/v1/openai/v1" not in openai_endpoint:
            openai_endpoint = openai_endpoint.rstrip("/")
            if openai_endpoint.endswith("/v1"):
                openai_endpoint = openai_endpoint[:-3]
            openai_endpoint = f"{openai_endpoint}/v1/openai/v1"

        openai_endpoint = openai_endpoint.rstrip("/")

        # Initialize LLM with OpenAI-compatible endpoint
        llm = ChatOpenAI(
            model=config.get("model", "llama-4-scout-17b-16e-w4a16"),
            api_key=config.get("api_key", "fake"),
            base_url=openai_endpoint,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 4096),
        )

        # Get MCP server URL for the tool group
        mcp_server_url = _get_mcp_server_url(tool_group, config)
        if not mcp_server_url:
            raise ValueError(f"No MCP server URL configured for tool_group: {tool_group}")

        # Initialize MCP client
        server_name = tool_group.replace("mcp::", "")
        mcp_client = MultiServerMCPClient(
            {
                server_name: {
                    "transport": "sse",
                    "url": mcp_server_url,
                }
            }
        )

        # Get tools from MCP server with error handling
        tools = await mcp_client.get_tools()
        console.print(f"[cyan]Loaded {len(tools)} tools from MCP server: {server_name}[/cyan]")

        # Create LangChain 1.0 agent
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

        # Log API request details
        logger.info("=== Llama-Stack API Request ===")
        if node_name:
            logger.info(f"Node Name: {node_name}")
        if system_prompt_id:
            logger.info(f"System Prompt ID: {system_prompt_id}")
        if user_prompt_id:
            logger.info(f"User Prompt ID: {user_prompt_id}")
        logger.info(f"API Endpoint: {openai_endpoint}")
        logger.info(f"Model: {config.get('model', 'N/A')}")
        logger.info(f"Tool Group: {tool_group}")
        logger.info(f"Raw Query (System Prompt):\n{system_prompt}")
        logger.info(f"Raw Query (User Prompt):\n{user_prompt}")

        # Invoke agent
        result = await agent.ainvoke({"messages": [{"role": "user", "content": user_prompt}]})

        # Extract final response from messages
        final_response = ""
        raw_messages = []
        all_ai_messages = []
        tool_messages = []

        for message in result.get("messages", []):
            if isinstance(message, HumanMessage):
                raw_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                content = message.content or ""
                tool_calls = getattr(message, "tool_calls", None)
                raw_messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                all_ai_messages.append(message)
            elif isinstance(message, ToolMessage):
                tool_messages.append(message)
                raw_messages.append({"role": "tool", "content": message.content})

        # Find the final response (last AI message without tool calls)
        for message in reversed(all_ai_messages):
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls and len(tool_calls) > 0:
                continue
            if message.content:
                final_response = message.content
                break

        tool_executed = len(tool_messages) > 0

        logger.info(f"Final Answer:\n{final_response}")
        logger.info(f"Tool Executed: {tool_executed}, Tool Message Count: {len(tool_messages)}")
        logger.info("=== Llama-Stack API Response End ===")

        return {
            "success": True,
            "final_answer": final_response,
            "raw_response": json.dumps([msg for msg in raw_messages], indent=2),
            "session_id": None,
            "messages": raw_messages,
            "tool_executed": tool_executed,
            "tool_message_count": len(tool_messages),
        }

    except Exception as e:
        error_msg = f"LangChain Agent failed ({type(e).__name__}): {str(e)}"
        console.print(f"[red]❌ {error_msg}[/red]")
        import traceback

        console.print("[dim]Full traceback:[/dim]")
        for line in traceback.format_exception(type(e), e, e.__traceback__)[-10:]:
            console.print(f"[dim]{line.rstrip()}[/dim]")
        return {
            "success": False,
            "error": error_msg,
            "final_answer": None,
            "raw_response": None,
            "session_id": None,
        }


def run_agent(
    system_prompt: str,
    user_prompt: str,
    tool_group: str,
    config: Dict[str, Any],
    max_infer_iters: int = 20,
    node_name: str = None,
    system_prompt_id: str = None,
    user_prompt_id: str = None,
) -> Dict[str, Any]:
    """
    Run a LangChain 1.0 agent with OpenAI-compatible LLM and MCP tools.

    max_infer_iters is kept only for compatibility with other helpers.
    """
    _ = max_infer_iters
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(_run_agent_async(system_prompt, user_prompt, tool_group, config))
        return loop.run_until_complete(
            _run_agent_async(system_prompt, user_prompt, tool_group, config, node_name, system_prompt_id, user_prompt_id)
        )
    except RuntimeError:
        return asyncio.run(
            _run_agent_async(system_prompt, user_prompt, tool_group, config, node_name, system_prompt_id, user_prompt_id)
        )

