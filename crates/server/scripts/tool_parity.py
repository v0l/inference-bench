import json
import sys
from openai import OpenAI

url, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
client = OpenAI(base_url=url, api_key="x", timeout=600)

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "add", "description": "Add two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}},
]
PRIOR = [
    {"role": "user", "content": "weather in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "c0", "content": "sunny, 21C"},
]
USER_TURNS = ["and Dallas for 3 days?"] + [
    "weather in Tokyo for 2 days", "what is 17 plus 25?", "add 3.5 and 4.25", "weather in Berlin",
    "weather in Lima, 5 days please", "sum of 100 and 250", "what's the weather like in Oslo?", "add 7 and -2",
    "forecast for Cairo for 4 days", "compute 12 + 30", "weather in Sydney", "weather in New York, 1 day",
    "add 0.1 and 0.2", "weather in Mumbai for 7 days", "what is 999 plus 1?", "weather in Rome",
    "weather in Madrid, 3 days", "add 21 and 21", "weather in Seoul for 6 days",
]
CANNED = {"get_weather": "rainy, 12C", "add": "42"}


def call(messages):
    r = client.chat.completions.create(model=model, messages=messages, tools=TOOLS, temperature=0, max_tokens=1500, seed=1)
    m = r.choices[0].message
    calls = [{"name": t.function.name, "args": json.loads(t.function.arguments)} for t in (m.tool_calls or [])]
    return r.choices[0].finish_reason, calls, m


turns = []
for u in USER_TURNS:
    finish, calls, _ = call(PRIOR + [{"role": "user", "content": u}])
    turns.append({"user": u, "finish": finish, "calls": calls})
    print(u, finish, calls, flush=True)

loop = []
messages = [{"role": "user", "content": "What is the weather in Paris, then add 2 and 3?"}]
for _ in range(3):
    finish, calls, m = call(messages)
    loop.append({"finish": finish, "calls": calls})
    messages.append(m.model_dump(exclude_none=True))
    if not m.tool_calls:
        break
    for t in m.tool_calls:
        messages.append({"role": "tool", "tool_call_id": t.id, "content": CANNED.get(t.function.name, "?")})
print("loop", loop, flush=True)
json.dump({"turns": turns, "loop": loop}, open(out, "w"), indent=1)
