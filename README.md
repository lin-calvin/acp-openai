# acp-openai-middleware

ACP-to-OpenAI compatible middleware — let OpenAI clients talk to ACP agents.

Mount any [ACP](https://agentclientprotocol.com) agent behind an OpenAI-compatible HTTP API,
so standard OpenAI SDKs and tools can use ACP agents seamlessly.

## Quick Start

```bash
pip install acp-openai-middleware
```

Launch the middleware pointing at your ACP agent:

```bash
acp-openai-middleware \
  --agent python \
  --agent-args "my_agent.py" \
  --api-key sk-my-key \
  --port 8000
```

Then use it with any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-my-key",
)

response = client.chat.completions.create(
    model="my_agent",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Streaming is supported:

```python
stream = client.chat.completions.create(
    model="my_agent",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
```

## How It Works

```
OpenAI Client ──HTTP──▶ FastAPI (/v1/chat/completions) ──▶ SessionPool ──▶ AgentManager ──stdio──▶ ACP Agent
```

1. **Agent Manager** spawns your ACP agent as a subprocess (one per API key namespace)
2. **Session Pool** uses longest-prefix matching to map stateless OpenAI chat histories to stateful ACP sessions
3. **Mapper** converts between OpenAI message format and ACP ContentBlocks
4. **Stream Adapter** maps ACP `session/update` notifications to OpenAI SSE `delta` chunks

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--agent` | (required) | Command to launch the ACP agent |
| `--agent-args` | `[]` | Additional arguments for the agent |
| `--agent-cwd` | `.` | Working directory for the agent |
| `--agent-env` | `[]` | `KEY=VALUE` environment variables for the agent |
| `--api-key` | (required) | Allowed API key (repeatable; empty = allow all) |
| `--port` | `8000` | HTTP listen port |
| `--host` | `0.0.0.0` | HTTP bind address |
| `--session-ttl` | `3600` | Seconds before idle session eviction |
| `--max-sessions` | `50` | Max sessions per API key namespace |
| `--log-level` | `info` | `debug`, `info`, `warning`, or `error` |

### Examples

```bash
# Run a Python ACP agent
acp-openai-middleware --agent python --agent-args "my_agent.py" --api-key sk-local

# Run a compiled agent binary
acp-openai-middleware --agent /usr/local/bin/codex-agent --api-key sk-prod --port 8080

# Multiple API keys, custom session TTL
acp-openai-middleware \
  --agent uv --agent-args "run examples/agent.py" \
  --api-key sk-alice \
  --api-key sk-bob \
  --session-ttl 7200
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/v1/models` | GET | List available models |

## Session Management

ACP agents maintain stateful sessions with conversation history. OpenAI clients are stateless and
send the full message history in each request. The middleware bridges this gap with
**longest-prefix matching**:

1. When a request arrives with messages `[m0, m1, ..., mN]`, the pool finds the ACP session
   whose tracked history is the longest prefix of the incoming messages
2. Only the new messages (after the matched prefix) are forwarded to the agent
3. The agent's response is added to the session history

Sessions are evicted after the configured TTL (default 1 hour).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
