## Research Agent Builder (Builder) and Research Agent (Researcher)

### Purpose
This project dynamically creates a domain-tailored research agent (Researcher) using a prewritten LangGraph agent (Builder). The Researcher emulates a small research team: a PI/orchestrator for planning/reasoning and specialized colleagues that execute tools. The system prioritizes data management, tool compatibility, and safe, reproducible workflows.

### Components
- **src/agent.py**: Assembles the Builder workflow and compiles the LangGraph graph. Finalizes the Researcher specification and persists it to `agent_specs.json`.
- **src/builder_states.py**: Defines Pydantic and TypedDict schemas for inputs, shared state, graph planning output, and tool building state.
- **src/graph_builder.py**: Clarifies requirements with the user, plans the Researcher (PI system prompt, colleagues, tool implementations), and orchestrates tool building via Send API.
- **src/tool_builder.py**: Independent subgraph that implements and registers each requested tool, including code and a tool card stored under `generated_tools/`.
- **src/predefined_tools.json**: Registry of built-in, universally available tools (do not re-specify for colleagues).
- **src/graph_examples.json**: End-to-end examples demonstrating minimal setups and complex setups with custom tools.

### What’s pre-implemented vs dynamically created
- Pre-implemented
  - Builder graph nodes for: clarification, graph planning, tool building, and finalization.
  - Predefined tools available to all nodes by default (`predefined_tools.json`).
  - Tool Builder subgraph that outputs Python tool code and a JSON tool card.

- Dynamically created (per user requirements)
  - Researcher PI system prompt (the orchestrator’s behavior).
  - Colleagues: name, role, expertise, system prompt, and tool list.
  - Custom tools: implemented and registered under `generated_tools/` with code and metadata.

### Data and tool schema (core rules)
- **Data management**
  - Raw data creation/deletion is reserved for a prewritten Data Formatter node in the Researcher.
  - All other nodes may only read/write/delete the data they create within their own node scope.
  - A single shared PyArrow database is used; datasets are referenced by string keys only.

- **Configuration**
  - Agent-wide configuration (GPU, APIs, env) lives in a config object passed into tools. Tools must treat it as read-only context.

- **Tool interface**
  - Required signature pattern: `def tool_name(*, db=database, config=config) -> str`
  - Tools access data only via `db` lookups by dataset name; they must save results back under clear, well-named keys.
  - Tools return a descriptive string (not large data objects) and must handle errors gracefully by returning informative messages.
  - Tools are wrapped when bound to LLMs so `db` and `config` aren’t exposed as user-facing parameters.

### Current robustness mechanisms
- Strong typing and structured output for planning and tool generation (Pydantic schemas).
- Predefined tool registry to reduce duplication and enforce common capabilities.
- Tool Builder enforces signature, DB usage, string-only return, and error-handling narrative.
- Tool registration persists code and a JSON tool card (name, description, examples, data inputs/outputs) for discoverability.

### Suggested schema-level enhancements
1. Database namespacing and provenance
   - Enforce dataset naming: `<node>/<stage>/<artifact>` (e.g., `ml_specialist/predictions/gp_v1`).
   - Maintain a metadata table (dataset registry) with creator node, lineage, schema hash, created_at, and version. Validate save/delete against this table to enforce ownership and allowed operations.

2. Scoped DB facade for tools
   - Pass a wrapped `db` that checks ACLs against the metadata table so a node can only mutate its own artifacts and can only read whitelisted inputs. This preserves the single-DB model while enforcing boundaries.

3. Schema validation layer
   - Define lightweight schema specs for common artifacts (e.g., predictions require columns: original keys, `mean`, `std`). Provide a helper: `validate_table(expected_schema|columns)` that tools call before saving.

4. Standardized result contracts
   - Keep return type `str`, and also persist a machine-readable status record per run (e.g., `runs/<tool>/<timestamp>` with status, metrics, and references to outputs). This enables reliable downstream parsing without changing the textual return contract.

5. Config typing and scoping
   - Use a typed config model (Pydantic) and pass a read-only/scoped view to tools. Disallow mutation and prevent accidental secret leakage in returned strings.

6. Tool versioning
   - Store tools as `<name>@<semver>.py` and add version into the tool card. Allow colleagues to pin versions, easing upgrades.

7. Concurrency-safe state
   - Continue using reducers (`operator.add`) for list-aggregate state (e.g., `completed_tool`) and avoid shared-mutation keys without reducers when using Send/parallelism.

### High-level flow (intended)
1. Clarify requirements with the user.
2. Plan the Researcher: PI prompt, colleagues, and any non-predefined tools to implement.
3. Build custom tools (in parallel via Send) and register them under `generated_tools/`.
4. Finalize the Researcher specification, including colleague configs and tool registry path.

### Notes
- All tools should protect users from silent failures: validate inputs, explain assumptions, and provide actionable guidance in error messages.
- Wrappers that bind tools to LLM nodes must hide `db` and `config`, providing only user-meaningful arguments (dataset keys and parameters).

### Repo paths of interest
- Builder/Researcher logic: `src/agent.py`, `src/graph_builder.py`, `src/builder_states.py`
- Tool builder and registry: `src/tool_builder.py`, `generated_tools/`
- Predefined tools and examples: `src/predefined_tools.json`, `src/graph_examples.json`


