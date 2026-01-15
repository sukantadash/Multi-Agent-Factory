#!/usr/bin/env python3
"""
Intelligent Operations Agent (Phase-1) - Refactored
A LangGraph-based workflow for automated error analysis, resolution discovery, and incident creation.

This implementation loads workflow configuration from JSON files:
- workflow_nodes_rules.json: Node definitions and routing logic
- workflow_prompts.json: System and user prompts
"""

import os
import json
import re
import logging
from typing import Annotated, Literal, TypedDict, Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# LangGraph imports
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# Agent helper imports
# Supported engines:
# - openai-compatible: OpenAI-compatible Chat Completions API via LangChain + MCP adapters
# - responses-api: OpenAI-compatible Responses API
load_dotenv('config.env')

# Engine selection (single flag)
AGENT_ENGINE = os.getenv('AGENT_ENGINE', 'responses-api').strip().lower()

def _load_agent_helper(engine: str):
    engine = (engine or '').strip().lower()
    if engine == 'responses-api':
        from agent_helper_responseapi import run_agent, extract_json_from_response
        print("[INFO] Using Responses API agent helper (agent_helper_responseapi)")
        return run_agent, extract_json_from_response

    if engine == 'openai-compatible':
        from agent_helper_openai import run_agent, extract_json_from_response
        print("[INFO] Using OpenAI-compatible agent helper (agent_helper_openai)")
        return run_agent, extract_json_from_response

    raise ValueError(f"Unsupported AGENT_ENGINE='{engine}'. Use 'responses-api' or 'openai-compatible'.")

run_agent, extract_json_from_response = _load_agent_helper(AGENT_ENGINE)


# ============================================================================
# Configuration Loaders
# ============================================================================

def load_workflow_config(rules_file: str = "workflow_nodes_rules.json") -> Dict[str, Any]:
    """Load workflow node rules from JSON file"""
    rules_path = Path(__file__).parent / rules_file
    with open(rules_path) as f:
        return json.load(f)


def load_prompts_config(prompts_file: str = "workflow_prompts.json") -> Dict[str, Any]:
    """Load prompts from JSON file"""
    prompts_path = Path(__file__).parent / prompts_file
    with open(prompts_path) as f:
        return json.load(f)


def load_config():
    """Load configuration from config.env file"""
    load_dotenv('config.env')
    temperature = float(os.getenv('TEMPERATURE', '0.7'))
    if temperature <= 0:
        temperature = 0.7
    
    openai_endpoint = os.getenv('OPENAI_ENDPOINT')
    base_url = os.getenv('LLAMA_STACK_URL', 'http://localhost:8321')
    
    if base_url and not base_url.startswith(('http://', 'https://')):
        base_url = f'https://{base_url}'
    
    if openai_endpoint:
        endpoint = openai_endpoint
    else:
        endpoint = base_url.rstrip('/')
        endpoint = f"{endpoint}/v1/openai/v1"
    
    endpoint = endpoint.rstrip('/')
    
    mcp_servers = {
        'openshift': os.getenv('MCP_OPENSHIFT_SERVER_URL', 'http://ocp-mcp-server:8000/sse'),
        'atlassian': os.getenv('MCP_ATLASSIAN_SERVER_URL', 'http://atlassian-mcp-server:8080/sse'),
    }
    
    slack_url = os.getenv('MCP_SLACK_SERVER_URL')
    if slack_url:
        mcp_servers['slack'] = slack_url
    
    return {
        'openai_endpoint': endpoint,
        'base_url': endpoint,
        'model': os.getenv('LLM_MODEL_ID', 'llama-4-scout-17b-16e-w4a16'),
        'api_key': os.getenv('API_KEY', 'fake'),
        'temperature': temperature,
        'max_tokens': int(os.getenv('MAX_TOKENS', '4096')),
        'timeout': int(os.getenv('LLAMA_STACK_TIMEOUT', '60')),
        'mcp_servers': mcp_servers,
        'namespace': os.getenv('DEFAULT_NAMESPACE', 'oom-test'),
        'jira_project_key': os.getenv('JIRA_PROJECT_KEY', 'OPS'),
        'jira_issue_type': os.getenv('JIRA_ISSUE_TYPE', 'Incident'),
        'confluence_space_key': os.getenv('CONFLUENCE_SPACE_KEY', 'OPS'),
    }


# ============================================================================
# State Definition
# ============================================================================

class WorkflowState(TypedDict):
    """State schema for the workflow graph"""
    messages: Annotated[list, add_messages]
    namespace: str
    errors_found: bool
    errors: list
    resolution_found: bool
    resolution_data: dict
    ai_resolution: dict
    confluence_page_url: str
    jira_ticket_key: str
    workflow_status: str
    error_message: str


# ============================================================================
# Prompt Manager
# ============================================================================

class PromptManager:
    """Manages prompt loading and rendering"""
    
    def __init__(self, prompts_config: Dict[str, Any]):
        self.prompts_config = prompts_config
        self.system_prompts = {p['prompt_id']: p for p in prompts_config['system_prompts']}
        self.user_prompts = {p['prompt_id']: p for p in prompts_config['user_prompts']}
    
    def get_prompt(self, prompt_id: str, prompt_type: str) -> Dict[str, Any]:
        """Get a prompt by ID and type"""
        if prompt_type == 'system':
            return self.system_prompts.get(prompt_id)
        else:
            return self.user_prompts.get(prompt_id)
    
    def render_prompt(self, prompt_template: str, context: Dict[str, Any]) -> str:
        """Render a prompt template with variables from context"""
        try:
            return prompt_template.format(**context)
        except KeyError as e:
            # If variable is missing, use empty string
            missing_var = str(e).strip("'")
            return prompt_template.replace(f"{{{missing_var}}}", "")
    
    def get_rendered_prompts(self, node_config: Dict[str, Any], state: WorkflowState, config: Dict[str, Any]) -> tuple[str, str]:
        """Get rendered system and user prompts for a node"""
        system_prompt_id = node_config.get('system_prompt_id')
        user_prompt_id = node_config.get('user_prompt_id')
        
        system_prompt_data = self.get_prompt(system_prompt_id, 'system')
        user_prompt_data = self.get_prompt(user_prompt_id, 'user')
        
        if not system_prompt_data or not user_prompt_data:
            raise ValueError(f"Prompts not found: SP={system_prompt_id}, UP={user_prompt_id}")
        
        # Extract variables from state/config based on input_parameters
        context = self._extract_context(node_config.get('input_parameters', []), state, config)
        
        system_prompt = self.render_prompt(system_prompt_data['prompt_template'], context)
        user_prompt = self.render_prompt(user_prompt_data['prompt_template'], context)
        
        return system_prompt, user_prompt
    
    def _extract_context(self, input_parameters: List[Dict[str, Any]], state: WorkflowState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context variables from state/config based on input_parameters
        
        This method dynamically extracts all input parameters for a node, including:
        - Direct values from state/config via source
        - Computed values via transform expressions
        - Default values when source is not available
        """
        context = {}
        
        # First pass: extract all direct values from source
        for param in input_parameters:
            param_name = param['name']
            source = param.get('source', '')
            default = param.get('default')
            transform = param.get('transform')
            
            # If there's a transform but no source, skip source resolution (transform will compute it)
            if transform and not source:
                continue
            
            # Parse source like "state.namespace" or "config['namespace']" or "state.errors[0].error_type"
            value = self._resolve_source(source, state, config, default)
            
            if value is not None:
                context[param_name] = value
        
        # Second pass: apply transforms (transforms may depend on other context values)
        for param in input_parameters:
            param_name = param['name']
            transform = param.get('transform')  # Optional transform function/expression
            
            # Apply transform if specified
            if transform:
                # For transforms, use existing context value if available, otherwise None
                value = context.get(param_name)
                transformed_value = self._apply_transform(value, transform, context, state, config)
                context[param_name] = transformed_value
        
        return context
    
    def _resolve_source(self, source: str, state: WorkflowState, config: Dict[str, Any], default: Any = None) -> Any:
        """Resolve a source expression like 'state.namespace' or 'config[\'namespace\']' or 'state.errors[0].error_type'"""
        if not source:
            return default
        
        try:
            # Handle state.* references
            if source.startswith('state.'):
                path = source[6:]  # Remove 'state.'
                return self._resolve_path(state, path)
            
            # Handle config['key'] references
            elif source.startswith('config['):
                # Extract key from config['key']
                match = re.search(r"config\['([^']+)'\]", source)
                if match:
                    key = match.group(1)
                    return config.get(key, default)
            
            # Handle computed values like f'{error_type} - {namespace}'
            elif source.startswith("f'") or source.startswith('f"'):
                # This is a computed value, will be handled in _extract_context
                return None
            
            # Handle 'formatted from ...' - special case
            elif 'formatted from' in source:
                return None  # Will be handled specially in _extract_context
            
            return default
        except Exception:
            return default
    
    def _resolve_path(self, obj: Any, path: str) -> Any:
        """Resolve a path like 'namespace' or 'errors[0].error_type'
        
        Handles:
        - Simple keys: 'namespace'
        - Nested keys: 'errors[0].error_type'
        - Array access: 'errors[0]'
        """
        if not path:
            return obj
        
        try:
            # Split by both . and [ but keep the delimiters for context
            # Use a more sophisticated approach to handle nested paths
            result = obj
            
            # Handle array access pattern: key[index].nested_key
            # Split by . first, then handle [index] in each part
            path_parts = path.split('.')
            
            for part in path_parts:
                if not part:
                    continue
                
                # Check if this part has array access like 'errors[0]'
                if '[' in part and ']' in part:
                    # Extract key and index: 'errors[0]' -> key='errors', index=0
                    match = re.match(r'^([^\[]+)\[(\d+)\]$', part)
                    if match:
                        key = match.group(1)
                        index = int(match.group(2))
                        
                        # Get the dict/list value
                        if isinstance(result, dict):
                            result = result.get(key)
                        else:
                            return None
                        
                        # Access array element
                        if isinstance(result, (list, tuple)) and 0 <= index < len(result):
                            result = result[index]
                        else:
                            return None
                    else:
                        return None
                else:
                    # Simple key access
                    if isinstance(result, dict):
                        result = result.get(part)
                    else:
                        return None
                
                if result is None:
                    return None
            
            return result
        except Exception:
            return None
    
    def _apply_transform(self, value: Any, transform: str, context: Dict[str, Any], state: WorkflowState, config: Dict[str, Any]) -> Any:
        """Apply a transform expression to a value
        
        Supports Python f-string expressions that can reference context variables.
        Example: f"{error_type} - {namespace}" or f"{context.get('key', 'default')}"
        """
        try:
            if transform.startswith('f"') or transform.startswith("f'"):
                # Python f-string expression - evaluate safely
                # Create safe evaluation context with all available variables
                # Include 'context' as a variable so transforms can use context.get()
                safe_dict = {
                    **context,  # Spread context keys for direct access
                    'context': context,  # Also include context as a dict for context.get() calls
                    'value': value,
                    'state': state,
                    'config': config,
                    'datetime': datetime
                }
                # Only allow safe operations
                safe_builtins = {
                    'str': str, 'int': int, 'float': float, 'bool': bool, 'len': len,
                    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                    'chr': chr, 'ord': ord, 'format': format, 'join': str.join
                }
                return eval(transform, {"__builtins__": safe_builtins}, safe_dict)
            else:
                # Unknown transform, return value as-is
                return value
        except Exception as e:
            # If transform fails, return original value
            logger.warning(f"Transform evaluation failed: {e}. Transform: {transform[:100]}")
            return value


# ============================================================================
# Node Executor
# ============================================================================

class NodeExecutor:
    """Executes workflow nodes based on configuration"""
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
        self.console = Console()
    
    def create_node_function(self, node_config: Dict[str, Any]) -> callable:
        """Create a node function from configuration"""
        node_name = node_config['node_name']
        node_display_name = node_config.get('node_display_name', node_name)
        
        def node_function(state: WorkflowState) -> WorkflowState:
            """Dynamic node function"""
            self.console.print(f"[bold blue]🔍 {node_display_name}[/bold blue]")
            self.console.print("=" * 70)
            
            # Generic validation based on input_parameters requirements
            validation_result = self._validate_input_parameters(node_config, state)
            if validation_result:
                return validation_result
            
            config = load_config()
            
            # Get rendered prompts
            try:
                system_prompt, user_prompt = self.prompt_manager.get_rendered_prompts(
                    node_config, state, config
                )
            except Exception as e:
                self.console.print(f"❌ Error loading prompts: {e}")
                return {
                    **state,
                    "workflow_status": "error",
                    "error_message": f"Error loading prompts: {e}"
                }
            
            # Get prompt IDs for logging
            system_prompt_id = node_config.get('system_prompt_id', 'N/A')
            user_prompt_id = node_config.get('user_prompt_id', 'N/A')
            
            # Log node name, prompt IDs, and raw query
            logger.info(f"=== Node Execution Start ===")
            logger.info(f"Node Name: {node_name}")
            logger.info(f"System Prompt ID: {system_prompt_id}")
            logger.info(f"User Prompt ID: {user_prompt_id}")
            logger.info(f"Raw Query (System Prompt):\n{system_prompt}")
            logger.info(f"Raw Query (User Prompt):\n{user_prompt}")
            
            # Execute agent
            result = run_agent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tool_group=node_config.get('tool_group', ''),
                config=config,
                max_infer_iters=node_config.get('max_infer_iters', 30),
                node_name=node_name,
                system_prompt_id=system_prompt_id,
                user_prompt_id=user_prompt_id
            )
            
            # Log raw response
            if result.get('success'):
                raw_response = result.get('raw_response', '')
                final_answer = result.get('final_answer', '')
                logger.info(f"Raw Response (JSON):\n{raw_response}")
                logger.info(f"Final Answer:\n{final_answer}")
            else:
                error = result.get('error', 'Unknown error')
                logger.error(f"Agent execution failed: {error}")
            
            logger.info(f"=== Node Execution End ===")
            
            # Process result
            if result['success']:
                return self._process_success_result(node_config, state, result)
            else:
                return self._process_error_result(node_config, state, result)
        
        return node_function
    
    def _validate_input_parameters(self, node_config: Dict[str, Any], state: WorkflowState) -> Optional[WorkflowState]:
        """Validate required input parameters before executing node"""
        input_params = node_config.get('input_parameters', [])
        missing_params = []
        
        # Load config for validation
        config = load_config()
        
        for param in input_params:
            if param.get('required', False):
                param_name = param['name']
                source = param.get('source', '')
                transform = param.get('transform')
                
                # Skip validation for computed values (transform only, no source)
                if transform and not source:
                    # Computed values will be generated, so skip validation
                    continue
                
                # Resolve value from state/config (use actual config, not empty dict)
                value = self.prompt_manager._resolve_source(source, state, config, param.get('default'))
                
                # Check if value is empty/None
                if value is None or value == "" or (isinstance(value, list) and len(value) == 0) or (isinstance(value, dict) and len(value) == 0):
                    missing_params.append(param_name)
        
        if missing_params:
            self.console.print(f"[yellow]⚠️  Missing required parameters: {', '.join(missing_params)}[/yellow]")
            
            # Set default error state based on output parameters
            error_state = {**state, "workflow_status": "error"}
            error_state["error_message"] = f"Missing required parameters: {', '.join(missing_params)}"
            
            # Set default values for output parameters
            for output_param in node_config.get('output_parameters', []):
                param_name = output_param['name']
                if param_name not in error_state:
                    param_type = output_param.get('type', 'str')
                    if param_type == 'bool':
                        error_state[param_name] = False
                    elif param_type == 'list':
                        error_state[param_name] = []
                    elif param_type == 'dict':
                        error_state[param_name] = {}
                    else:
                        error_state[param_name] = ""
            
            return error_state
        
        return None
    
    def _process_success_result(self, node_config: Dict[str, Any], state: WorkflowState, result: Dict[str, Any]) -> WorkflowState:
        """Process successful agent result"""
        final_answer = result.get('final_answer', '')
        tool_executed = result.get('tool_executed', False)
        tool_message_count = result.get('tool_message_count', 0)
        
        if not tool_executed and tool_message_count == 0:
            self.console.print(f"[yellow]⚠️  Warning: Tools were not executed (tool_message_count: {tool_message_count})[/yellow]")
        
        # Extract JSON from response
        try:
            response_data = extract_json_from_response(final_answer)
            if response_data and isinstance(response_data, dict):
                # Map response data to output parameters dynamically
                updated_state = {**state}
                
                # Extract all output parameters from response_data
                for output_param in node_config.get('output_parameters', []):
                    param_name = output_param['name']
                    
                    # Try to get value from response_data
                    if param_name in response_data:
                        updated_state[param_name] = response_data[param_name]
                    else:
                        # If not in response, set default based on type
                        param_type = output_param.get('type', 'str')
                        if param_type == 'bool':
                            updated_state[param_name] = False
                        elif param_type == 'list':
                            updated_state[param_name] = []
                        elif param_type == 'dict':
                            updated_state[param_name] = {}
                        else:
                            updated_state[param_name] = ""
                
                # Get node display name early (used in multiple places)
                node_display_name = node_config.get('node_display_name', node_config['node_name'])
                
                # Set workflow_status based on response or default
                # Check if response_data has a status indicator (like 'page_found', 'ticket_created', etc.)
                # or use a generic success status
                if 'workflow_status' not in updated_state:
                    # Determine status from common patterns in response
                    # Check for common success indicators
                    if any(key in response_data for key in ['page_found', 'ticket_created', 'page_created', 'errors_found']):
                        # Use a generic success status
                        updated_state['workflow_status'] = f"{node_config['node_name']}_completed"
                    else:
                        updated_state['workflow_status'] = f"{node_config['node_name']}_completed"
                
                # Update messages
                if 'messages' in updated_state:
                    updated_state['messages'] = state.get('messages', []) + [
                        {"role": "assistant", "content": f"{node_display_name} completed"}
                    ]
                
                # Log success
                self.console.print(f"✅ {node_display_name} completed")
                
                # Log key output values for debugging
                key_outputs = [op['name'] for op in node_config.get('output_parameters', [])[:3]]  # First 3 outputs
                for key in key_outputs:
                    if key in updated_state and updated_state[key]:
                        if isinstance(updated_state[key], (list, dict)):
                            self.console.print(f"   {key}: {type(updated_state[key]).__name__} with {len(updated_state[key])} items")
                        elif isinstance(updated_state[key], bool):
                            self.console.print(f"   {key}: {updated_state[key]}")
                        elif isinstance(updated_state[key], str) and len(updated_state[key]) < 100:
                            self.console.print(f"   {key}: {updated_state[key]}")
                
                if tool_executed:
                    self.console.print(f"   Tools executed: {tool_message_count} tool calls")
                
                return updated_state
            else:
                # JSON parsing failed
                self.console.print("⚠️  Could not parse response as JSON")
                self.console.print(f"[dim]Response preview (first 500 chars):[/dim]")
                self.console.print(f"[dim]{final_answer[:500]}[/dim]")
                
                return {
                    **state,
                    "workflow_status": "error",
                    "error_message": f"Could not parse response. Response length: {len(final_answer)} chars"
                }
        except Exception as e:
            self.console.print(f"❌ Error parsing response: {e}")
            return {
                **state,
                "workflow_status": "error",
                "error_message": f"Error parsing response: {e}"
            }
    
    def _process_error_result(self, node_config: Dict[str, Any], state: WorkflowState, result: Dict[str, Any]) -> WorkflowState:
        """Process failed agent result"""
        error_msg = result.get('error', 'Unknown error')
        self.console.print(f"❌ {node_config.get('node_display_name', 'Node')} failed: {error_msg}")
        
        # Set default error state based on output parameters
        error_state = {**state, "workflow_status": "error", "error_message": error_msg}
        
        # Set default values for output parameters
        for output_param in node_config.get('output_parameters', []):
            param_name = output_param['name']
            if param_name not in error_state:
                param_type = output_param.get('type', 'str')
                if param_type == 'bool':
                    error_state[param_name] = False
                elif param_type == 'list':
                    error_state[param_name] = []
                elif param_type == 'dict':
                    error_state[param_name] = {}
                else:
                    error_state[param_name] = ""
        
        return error_state


# ============================================================================
# Graph Builder
# ============================================================================

class WorkflowGraphBuilder:
    """Builds LangGraph workflow from JSON configuration"""
    
    def __init__(self, workflow_config: Dict[str, Any], prompt_manager: PromptManager):
        self.workflow_config = workflow_config
        self.prompt_manager = prompt_manager
        self.node_executor = NodeExecutor(prompt_manager)
        self.node_functions = {}
    
    def build(self) -> Any:
        """Build and compile the workflow graph"""
        graph = StateGraph(WorkflowState)
        
        # Create node functions
        active_nodes = []
        for node_config in self.workflow_config['nodes']:
            node_name = node_config['node_name']
            if node_config.get('node_status') == 'active':
                node_func = self.node_executor.create_node_function(node_config)
                self.node_functions[node_name] = node_func
                graph.add_node(node_name, node_func)
                active_nodes.append(node_name)
        
        if not active_nodes:
            raise ValueError("No active nodes found in workflow configuration")
        
        # Set entry point
        start_node = next(
            (n['node_name'] for n in self.workflow_config['nodes'] if n.get('is_start_node')),
            None
        )
        if not start_node:
            raise ValueError("No start node found in workflow configuration")
        
        if start_node not in active_nodes:
            raise ValueError(f"Start node '{start_node}' is not active")
        
        graph.set_entry_point(start_node)
        
        # Add edges
        for node_config in self.workflow_config['nodes']:
            node_name = node_config['node_name']
            
            # Skip if node is not active
            if node_config.get('node_status') != 'active':
                continue
            
            # Check if node has both conditional and direct edges (invalid configuration)
            has_conditional = 'condition_field' in node_config
            has_direct = 'edge_node' in node_config
            
            if has_conditional and has_direct:
                logger.warning(f"Node {node_name} has both conditional and direct edges. Using conditional edge.")
            
            # Conditional edges
            if has_conditional:
                condition_field = node_config['condition_field']
                true_node = node_config.get('condition_true_node')
                false_node = node_config.get('condition_false_node')
                
                if not true_node or not false_node:
                    logger.error(f"Node {node_name} has condition_field but missing condition_true_node or condition_false_node")
                    continue
                
                def make_condition_func(field: str, true_target: str, false_target: str):
                    def condition_func(state: WorkflowState) -> str:
                        value = state.get(field, False)
                        return true_target if value else false_target
                    return condition_func
                
                condition_func = make_condition_func(condition_field, true_node, false_node)
                
                graph.add_conditional_edges(
                    node_name,
                    condition_func,
                    {
                        true_node: true_node,
                        false_node: false_node
                    }
                )
            
            # Direct edges
            elif has_direct:
                edge_node = node_config['edge_node']
                if edge_node:
                    graph.add_edge(node_name, edge_node)
                # If edge_node is empty or None, node is an end node (no edge added)
            
            # If node has neither conditional nor direct edges, it's an end node
            # LangGraph will handle this automatically
        
        return graph.compile()


# ============================================================================
# Main Execution
# ============================================================================

def display_workflow_summary(state: WorkflowState):
    """Display workflow summary"""
    console = Console()
    
    table = Table(title="Workflow Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Namespace", state.get("namespace", "Unknown"))
    table.add_row("Errors Found", str(state.get("errors_found", False)))
    table.add_row("Error Count", str(len(state.get("errors", []))))
    table.add_row("Resolution Found", str(state.get("resolution_found", False)))
    table.add_row("Confluence Page URL", state.get("confluence_page_url", "N/A"))
    table.add_row("Jira Ticket Key", state.get("jira_ticket_key", "N/A"))
    table.add_row("Workflow Status", state.get("workflow_status", "Unknown"))
    
    if state.get("error_message"):
        table.add_row("Error Message", state.get("error_message", ""))
    
    console.print(table)


def main(namespace=None):
    """Main execution function"""
    console = Console()
    console.print("[bold green]🚀 INTELLIGENT OPERATIONS AGENT (PHASE-1)[/bold green]")
    console.print("=" * 70)
    
    # Load configurations
    try:
        workflow_config = load_workflow_config()
        prompts_config = load_prompts_config()
    except Exception as e:
        console.print(f"[red]❌ Failed to load configuration: {e}[/red]")
        return None
    
    config = load_config()
    namespace = namespace or config['namespace']
    
    # Initialize state
    initial_state: WorkflowState = {
        "messages": [{"role": "user", "content": f"Analyze errors in namespace: {namespace}"}],
        "namespace": namespace,
        "errors_found": False,
        "errors": [],
        "resolution_found": False,
        "resolution_data": {},
        "ai_resolution": {},
        "confluence_page_url": "",
        "jira_ticket_key": "",
        "workflow_status": "initialized",
        "error_message": ""
    }
    
    # Build and run workflow
    console.print(f"\n[bold yellow]Building workflow graph from configuration...[/bold yellow]")
    
    prompt_manager = PromptManager(prompts_config)
    graph_builder = WorkflowGraphBuilder(workflow_config, prompt_manager)
    app = graph_builder.build()
    
    console.print(f"\n[bold yellow]Executing workflow for namespace: {namespace}[/bold yellow]")
    console.print("-" * 70)
    
    try:
        final_state = app.invoke(initial_state)
        
        console.print("\n" + "=" * 70)
        console.print("[bold green]✅ WORKFLOW COMPLETED![/bold green]")
        console.print("=" * 70)
        
        display_workflow_summary(final_state)
        
        return final_state
        
    except Exception as e:
        console.print(f"\n[red]❌ Workflow failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {
            **initial_state,
            "workflow_status": "failed",
            "error_message": str(e)
        }


if __name__ == "__main__":
    import sys
    
    # Get namespace from command line argument or use default
    namespace = sys.argv[1] if len(sys.argv) > 1 else None
    
    result = main(namespace)
    
    # Exit with appropriate code
    if result and result.get("workflow_status") == "complete":
        sys.exit(0)
    else:
        sys.exit(1)
