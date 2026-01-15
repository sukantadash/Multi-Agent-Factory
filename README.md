# High-Level Design — Multi‑Agent Factory (Config‑Driven Workflow)

This project implements a **multi-agent factory**: you can build *any* use case by composing a workflow from configurable “agents” (nodes), each with its own tools and prompts, and routing logic that orchestrates them.

## LLM runtime / agent engines (llama-stack 0.3.0)

This factory is designed for **llama-stack 0.3.0** and can be configured to use two agent engines:

- **OpenAI-compatible APIs**: `agent_helper_openai.py` (Chat Completions via OpenAI-compatible endpoint + MCP adapters)
- **Responses APIs**: `agent_helper_responseapi.py` (Responses API via OpenAI-compatible endpoint)

Select the engine in `config.env`:

- `AGENT_ENGINE=openai-compatible`
- `AGENT_ENGINE=responses-api`

At runtime, `agent_workflow.py` reads two JSON configuration files:

- `workflow_nodes_rules.json`: **orchestration** (graph structure + node I/O contract + routing).
- `workflow_prompts.json`: **agent behavior** (system/user prompt templates), referenced by ID from nodes.

## Core concepts

### Workflow = Orchestrator (LangGraph)

- The workflow is a **LangGraph StateGraph**.
- A workflow runs by repeatedly executing nodes (agents) and updating a shared **state** object (`WorkflowState`).
- Routing between nodes is handled by:
  - **conditional edges** (branch based on a boolean field in state), or
  - **direct edges** (always go to a fixed next node).

### Node = Agent (specialized role + tools + I/O)

Each node is a **specialized agent instance** created from configuration:

- **Persona + task**: selected by `system_prompt_id` and `user_prompt_id`.
- **Tool access**: controlled by `tool_group` (e.g., OpenShift vs Atlassian tools).
- **Input contract**: `input_parameters[]` declares which values are injected into prompt templates.
- **Output contract**: `output_parameters[]` declares which JSON keys are expected back from the node.

Practically, a node is “an agent with a job”:

- It receives state + config.
- It renders prompts using a computed context.
- It runs the LLM + tools.
- It returns JSON, which is merged into the workflow state.

### Prompts = reusable building blocks (decoupled from nodes)

`workflow_prompts.json` contains:

- `system_prompts[]`: global behavioral constraints (tool calling rules, formatting rules).
- `user_prompts[]`: task instruction + expected output schema.

Nodes reference prompts by ID, which enables:

- Reusing prompts across nodes.
- Swapping prompts without changing orchestration logic.
- Maintaining a library of prompts independent of workflow graphs.

## Configuration contracts (what must stay consistent)

### 1) Node → Prompt variable contract

Prompt templates use `{variable}` placeholders. Those variables must be provided by the node’s `input_parameters[]`.

Inputs can come from:

- `state.<field>` (supports simple indexing, e.g. `state.errors[0].error_type`)
- `config['key']` (values from `config.env`)
- `transform` (computed values via evaluated f-string expressions)

If a variable is missing, prompt rendering may produce an empty value, which can break the task or lead to ambiguous outputs.

### 2) Node output JSON → Workflow state contract

The workflow engine parses the node’s final answer as **JSON** and copies values into state based on `output_parameters[]`.

If the node returns JSON without the expected keys:

- routing decisions can be wrong (e.g., `condition_field` not set),
- downstream nodes may fail input validation (required inputs missing).

### 3) Tool group contract

The node’s `tool_group` must match what the prompts instruct the agent to call.

Example:

- If the user prompt tells the agent to call Confluence tools, the node must be `tool_group: "mcp::atlassian"`.

## Factory extension points (how to implement “any use case”)

To build a new use case, you typically only edit JSON:

- **Add/modify nodes** in `workflow_nodes_rules.json`
  - Choose tool group(s) per node
  - Declare inputs/outputs
  - Define routing (conditional or direct)
- **Add/modify prompts** in `workflow_prompts.json`
  - Define agent persona constraints (system prompts)
  - Define tasks and strict JSON output schema (user prompts)

This pattern supports many classes of workflows:

- **diagnose → search knowledge → generate remediation → publish → open ticket**
- **ingest → enrich → validate → summarize → notify**
- **triage → route → execute runbook → confirm**

## Example — Intelligent Operations Agent (Phase‑1)

The “Intelligent Operations Agent Phase‑1” is a concrete workflow defined by:

- `workflow_nodes_rules.json` (5 nodes)
- `workflow_prompts.json` (SP001–SP005, UP001–UP005)

### Workflow structure (nodes and routing)

1. **`identify_errors`** (OpenShift tools)
   - Purpose: inspect a namespace for failing pods/events/logs.
   - Outputs: `errors_found`, `errors`
   - Routing: if `errors_found == true` → `search_confluence`, else → `__end__`

2. **`search_confluence`** (Atlassian tools)
   - Purpose: search Confluence for a known resolution based on `error_type`.
   - Outputs: `resolution_found`, `resolution_data`
   - Routing: if `resolution_found == true` → `create_jira_ticket`, else → `generate_ai_resolution`

3. **`generate_ai_resolution`** (LLM reasoning)
   - Purpose: generate a remediation plan if no KB page exists.
   - Output: `ai_resolution`
   - Routing: direct → `save_to_confluence`

4. **`save_to_confluence`** (Atlassian tools)
   - Purpose: publish the AI-generated remediation as a Confluence page.
   - Output: `resolution_data` (now points to the created page), `resolution_found=true`
   - Routing: direct → `create_jira_ticket`

5. **`create_jira_ticket`** (Atlassian tools)
   - Purpose: open a Jira incident with error details + resolution reference.
   - Output: `jira_ticket_key`
   - Routing: direct → `__end__`

### Prompt mapping (node → prompt IDs)

The workflow uses prompt IDs from `workflow_prompts.json`:

- `identify_errors` → `SP001` + `UP001`
- `search_confluence` → `SP002` + `UP002`
- `generate_ai_resolution` → `SP003` + `UP003`
- `save_to_confluence` → `SP004` + `UP004`
- `create_jira_ticket` → `SP005` + `UP005`

### Why this demonstrates the “factory” model

- Each node is a different “agent persona” with different tools and strict JSON output.
- The orchestrator doesn’t hardcode task logic; it only:
  - renders prompts from inputs,
  - runs the agent with a tool group,
  - parses JSON into state,
  - routes based on state.

This means you can replace the entire business workflow (e.g., “Intelligent Operation Agent” or "Accelerating OCP-Virt Migration Project with AI") with a different one (e.g., “fraud investigation” or “loan onboarding verification”) by swapping the JSON configs while keeping the same execution engine.


