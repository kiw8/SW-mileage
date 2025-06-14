import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class Configuration:
    def __init__(self) -> None:
        self.load_env()
        self.slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.slack_app_token = os.getenv("SLACK_APP_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo")

    @staticmethod
    def load_env() -> None:
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        if "gpt" in self.llm_model.lower() and self.openai_api_key:
            return self.openai_api_key
        elif "llama" in self.llm_model.lower() and self.groq_api_key:
            return self.groq_api_key
        elif "claude" in self.llm_model.lower() and self.anthropic_api_key:
            return self.anthropic_api_key

        if self.openai_api_key:
            return self.openai_api_key
        elif self.groq_api_key:
            return self.groq_api_key
        elif self.anthropic_api_key:
            return self.anthropic_api_key

        raise ValueError("No API key found for any LLM provider")

class Server:
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config.get("env", {})},
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")

class Tool:
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def format_for_llm(self) -> str:
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

class LLMClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = 30.0
        self.max_retries = 2

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        if self.model.startswith("gpt-"):
            return await self._get_openai_response(messages)
        elif self.model.startswith("llama-"):
            return await self._get_groq_response(messages)
        elif self.model.startswith("claude-"):
            return await self._get_anthropic_response(messages)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    async def _get_openai_response(self, messages: List[Dict[str, str]]) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    if response.status_code == 200:
                        return response.json()["choices"][0]["message"]["content"]
                    if attempt == self.max_retries:
                        return f"Error: {response.status_code} - {response.text}"
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed: {str(e)}"
                await asyncio.sleep(2 ** attempt)

    async def _get_groq_response(self, messages: List[Dict[str, str]]) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    if response.status_code == 200:
                        return response.json()["choices"][0]["message"]["content"]
                    if attempt == self.max_retries:
                        return f"Error: {response.status_code} - {response.text}"
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed: {str(e)}"
                await asyncio.sleep(2 ** attempt)

    async def _get_anthropic_response(self, messages: List[Dict[str, str]]) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        system_message = None
        formatted_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                formatted_msgs.append({"role": msg["role"], "content": msg["content"]})
        payload = {
            "model": self.model,
            "messages": formatted_msgs,
            "temperature": 0.7,
            "max_tokens": 1500,
        }
        if system_message:
            payload["system"] = system_message
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    if response.status_code == 200:
                        return response.json()["content"][0]["text"]
                    if attempt == self.max_retries:
                        return f"Error: {response.status_code} - {response.text}"
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed: {str(e)}"
                await asyncio.sleep(2 ** attempt)

class SlackMCPBot:
    def __init__(self, slack_bot_token: str, slack_app_token: str, servers: List[Server], llm_client: LLMClient) -> None:
        self.app = AsyncApp(token=slack_bot_token)
        self.socket_mode_handler = AsyncSocketModeHandler(self.app, slack_app_token)
        self.client = AsyncWebClient(token=slack_bot_token)
        self.servers = servers
        self.llm_client = llm_client
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.tools: List[Tool] = []

        self.app.command("/tools")(self.show_tools_command)
        self.app.event("app_mention")(self.handle_mention)
        self.app.message()(self.handle_message)
        self.app.event("app_home_opened")(self.handle_home_opened)

    async def initialize_servers(self) -> None:
        for server in self.servers:
            try:
                await server.initialize()
                server_tools = await server.list_tools()
                self.tools.extend(server_tools)
                logging.info(f"Initialized server {server.name} with {len(server_tools)} tools")
            except Exception as e:
                logging.error(f"Failed to initialize server {server.name}: {e}")

    async def initialize_bot_info(self) -> None:
        try:
            auth_info = await self.client.auth_test()
            self.bot_id = auth_info["user_id"]
            logging.info(f"Bot initialized with ID: {self.bot_id}")
        except Exception as e:
            logging.error(f"Failed to get bot info: {e}")
            self.bot_id = None

    async def show_tools_command(self, ack, say, command):
        await ack()
        tools_list = "
".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        await say(f"🔧 사용 가능한 MCP 도구 목록:
{tools_list}")

    async def handle_mention(self, event, say):
        await self._process_message(event, say)

    async def handle_message(self, message, say):
        if message.get("channel_type") == "im" and not message.get("subtype"):
            await self._process_message(message, say)

    async def handle_home_opened(self, event, client):
        user_id = event["user"]
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "Welcome to MCP Assistant!"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "I'm an AI assistant with access to tools via MCP."}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "*Available Tools:*"}},
        ]
        for tool in self.tools:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"• *{tool.name}*: {tool.description}"}})
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "*How to Use:*
• Send a DM
• Mention @MCP Assistant
• Use /tools command"}})
        try:
            await client.views_publish(user_id=user_id, view={"type": "home", "blocks": blocks})
        except Exception as e:
            logging.error(f"Error publishing home view: {e}")

    async def _process_message(self, event, say):
        channel = event["channel"]
        user_id = event.get("user")
        if user_id == getattr(self, "bot_id", None):
            return
        text = event.get("text", "")
        if self.bot_id:
            text = text.replace(f"<@{self.bot_id}>", "").strip()
        thread_ts = event.get("thread_ts", event.get("ts"))
        if "도움말" in text:
            await say(text="도움말: @MCP Assistant로 호출하거나 DM을 보내세요. /tools 명령어도 사용 가능합니다.", channel=channel, thread_ts=thread_ts)
            return
        if channel not in self.conversations:
            self.conversations[channel] = {"messages": []}
        try:
            tools_text = "
".join([tool.format_for_llm() for tool in self.tools])
            system_message = {"role": "system", "content": f"You are an assistant. Tools:
{tools_text}
Use [TOOL] tool_name
{{...}} format."}
            self.conversations[channel]["messages"].append({"role": "user", "content": text})
            messages = [system_message] + self.conversations[channel]["messages"][-5:]
            response = await self.llm_client.get_response(messages)
            if "[TOOL]" in response:
                response = await self._process_tool_call(response, channel)
            self.conversations[channel]["messages"].append({"role": "assistant", "content": response})
            await say(text=response, channel=channel, thread_ts=thread_ts)
        except Exception as e:
            logging.error(f"Error processing message: {e}", exc_info=True)
            await say(text=f"⚠️ 오류 발생: {str(e)}", channel=channel, thread_ts=thread_ts)

    async def _process_tool_call(self, response: str, channel: str) -> str:
        try:
            tool_parts = response.split("[TOOL]")[1].strip().split("
", 1)
            tool_name = tool_parts[0].strip()
            if len(tool_parts) < 2:
                return f"'{tool_name}' 도구 실행 실패: JSON 인자 없음."
            try:
                arguments = json.loads(tool_parts[1].strip())
            except json.JSONDecodeError:
                return f"'{tool_name}' 도구의 JSON 인자가 잘못됨."
            for server in self.servers:
                server_tools = [tool.name for tool in await server.list_tools()]
                if tool_name in server_tools:
                    tool_result = await server.execute_tool(tool_name, arguments)
                    self.conversations[channel]["messages"].append({"role": "system", "content": f"Tool result:
{tool_result}"})
                    messages = [
                        {"role": "system", "content": "Interpret this tool result."},
                        {"role": "user", "content": f"{tool_name} result:
{tool_result}"}
                    ]
                    return await self.llm_client.get_response(messages)
            return f"'{tool_name}' 도구를 찾을 수 없습니다."
        except Exception as e:
            logging.error(f"Tool execution error: {e}")
            return f"도구 실행 중 오류: {str(e)}"

    async def start(self) -> None:
        await self.initialize_servers()
        await self.initialize_bot_info()
        logging.info("✅ Slack bot is running...")
        asyncio.create_task(self.socket_mode_handler.start_async())

    async def cleanup(self) -> None:
        try:
            if hasattr(self, "socket_mode_handler"):
                await self.socket_mode_handler.close_async()
        except Exception as e:
            logging.error(f"Handler close error: {e}")
        for server in self.servers:
            await server.cleanup()

async def main() -> None:
    config = Configuration()
    if not config.slack_bot_token or not config.slack_app_token:
        raise ValueError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set")
    server_config = config.load_config("servers_config.json")
    servers = [Server(name, conf) for name, conf in server_config["mcpServers"].items()]
    llm_client = LLMClient(config.llm_api_key, config.llm_model)
    bot = SlackMCPBot(config.slack_bot_token, config.slack_app_token, servers, llm_client)
    try:
        await bot.start()
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        await bot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
