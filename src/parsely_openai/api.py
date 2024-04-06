import datetime
import json
import logging
import os
import time

from typing import Dict, List

import openai
from tiktoken import encoding_for_model

_logger = logging.getLogger(__name__)

GPT4 = "gpt-4-0125-preview"
GPT4_TURBO = "gpt-4-turbo-preview"
GPT3 = "gpt-3.5-turbo-1106"

MAX_RETRIES = 3
BACKOFF_FACTOR = 2


def fetch_tools(toolkit):
    from parsely.tools import to_dict
    return [{"type": "function", "function": to_dict(func_def)} for func_def in toolkit]


def create_assistant(
    client: openai.Client, name: str, instructions: str, tools: List[Dict]
):
    return client.beta.assistants.create(
        instructions=instructions, name=name, tools=tools, model=GPT4
    )


class OpenAIThread:
    """Run using the OpenAI beta threads API"""

    POLL_RATE = 1.0

    def __init__(
        self, client, assistant_id, initial_prompt, files_attached=None, file_cache=None
    ):
        self.client = client
        self.assistant_id = assistant_id
        self.prompt = initial_prompt
        self.files_attached = files_attached
        self.file_cache = {} if file_cache is None else file_cache

        thread = self.client.beta.threads.create()
        self.thread_id = thread.id
        self.last_message = None
        self.last_time_failed = None
        self.n_restarts = 0

        self._start_messages()

        _logger.info("Thread: %s", self.thread_id)

    def _start_messages(self):
        file_ids = []
        if self.files_attached:
            for path in self.files_attached:
                if path in self.file_cache:
                    file_ids.append(self.file_cache[path])
                    continue

                file_obj = self.client.files.create(
                    file=path, purpose="assistants"
                )
                file_ids.append(file_obj.id)
                self.file_cache[path] = file_obj.id

        self.client.beta.threads.messages.create(
            self.thread_id, role="user", content=self.prompt, file_ids=file_ids
        )

    def messages(self, limit=100):
        messages = []
        after = self.last_message

        while True:
            response = self.client.beta.threads.messages.list(
                thread_id=self.thread_id, limit=limit, after=after, order="asc"
            )
            for item in response.data:
                after = item.id
                text = "\n".join([content.text.value for content in item.content])
                messages.append((item.role, text))

            if len(response.data) < limit:
                break

        self.last_message = after
        return messages

    def cleanup_files(self):
        file_ids = [self.file_cache.pop(key)
                    for key in list(self.file_cache.keys())]
        for file_id in file_ids:
            self.client.files.delete(file_id)

    def send_messages(self, tool_provider):
        if hasattr(tool_provider, "send_messages"):
            tool_provider.send_messages(self.messages())

    def handle_tool_call(self, run, tool_provider, responses: List):
        self.last_time_failed = None
        responses.clear()
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        # TODO: parallelize
        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.decoder.JSONDecodeError:
                # Assume the model was trying for "none"
                _logger.warn("Model gave invalid JSON input")
                arguments = {}

            _logger.info("Tool call: %s(%s)", tool_call.function.name, arguments)
            output = tool_provider(tool_call.function.name, arguments)
            _logger.info("Output: %s", output)
            responses.append(
                {"tool_call_id": tool_call.id, "output": json.dumps(output)}
            )

        unix_now = datetime.datetime.now().timestamp()
        if unix_now > run.expires_at:
            # Run has expired, we need to recreate it.
            run = self.submit_thread()

        while run.status in ("queued", "in_progress", "requires_action"):
            if run.status == "requires_action":
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                new_responses = []
                old_ids = {item["tool_call_id"]: item for item in responses}

                # The new tools calls might be fewer for some reason (??)
                if all(tool_call.id in old_ids for tool_call in tool_calls):
                    new_responses = [old_ids[tool_call.id] for tool_call in tool_calls]
                else:
                    # We have the match up the tool calls because they aren't
                    # stable. Try this and hope it works.
                    for tool_call, response in zip(tool_calls, responses):
                        new_responses.append(
                            {"tool_call_id": tool_call.id, "output": response["output"]}
                        )

                try:
                    return self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=self.thread_id,
                        run_id=run.id,
                        tool_outputs=new_responses,
                    )
                except openai.NotFoundError:
                    # Patching over more bullshit
                    run = self.handle_restart(run)
            else:
                time.sleep(self.POLL_RATE)
                run = self.client.beta.threads.runs.retrieve(
                    run_id=run.id, thread_id=self.thread_id
                )
                _logger.info("Polling new state %s", run.status)

        if run.status in ("completed", "failed", "cancelled"):
            return run
        _logger.error("Run is broken: %s", run)
        raise RuntimeError("They Fucked It Up Again")

    def handle_restart(self, run):
        _logger.info("Run expired, recreating")
        self.last_time_failed = "run expired"
        self.n_restarts += 1
        try:
            run = self.submit_thread()
            self.n_restarts = 0
        except openai.BadRequestError as err:
            _logger.warn("%s: %s", run, err)
            if self.n_restarts > 3:
                self.tool_provider.abort("Too many retries without a success")
                raise RuntimeError("Abort did not raise an exception")

            runs = self.client.beta.threads.runs.list(self.thread_id)
            for other_run in runs:
                if other_run.status not in (
                    "failed",
                    "completed",
                    "expired",
                    "cancelled",
                ):
                    _logger.info("Picked run: %s", other_run)
                    return other_run

            _logger.warn("Did not find another run")
            return run

    def handle_failure(self, run, responses, tool_provider):
        if self.last_time_failed:
            err = (
                f"Last failure: {self.last_time_failed}"
                f" this failure: {run.last_error}"
            )
            if tool_provider:
                tool_provider.abort(err)

        _logger.warn("Run %s failed due to %s. Retrying.", run.id, run.last_error)
        self.last_time_failed = f"Run failed due to {run.last_error}"
        return self.submit_thread()

    def add_chat(self, message: str):
        self.client.beta.threads.messages.create(
            self.thread_id, role="user", content=message
        )

    def run(self, tool_provider):
        """Runs the thread until complete"""
        run = self.submit_thread()
        _logger.info("Run: %s", run.id)
        responses = []
        self.last_time_failed = None

        while True:
            # Poll state
            if run is None:
                runs = self.client.beta.threads.runs.list(self.thread_id)
                for other_run in runs:
                    if other_run.status not in ("expired", "cancelled"):
                        run = other_run
                        break
                else:
                    run = self.submit_thread()
            else:
                run = self.client.beta.threads.runs.retrieve(
                    run_id=run.id, thread_id=self.thread_id
                )
            if run.status in "completed":
                break
            elif run.status == "requires_action":
                self.send_messages(tool_provider)
                run = self.handle_tool_call(run, tool_provider, responses)
            elif run.status == "cancelled":
                _logger.error("Run cancelled! %s", run)
                run = self.handle_restart(run)
            elif run.status == "expired":
                run = self.handle_restart(run)
            elif run.status == "failed":
                run = self.handle_failure(run, responses, tool_provider)
            else:
                self.last_time_failed = None
                self.send_messages(tool_provider)
            time.sleep(self.POLL_RATE)

        if tool_provider:
            tool_provider.complete("Task complete")
        _logger.info("Complete")

    def submit_thread(self):
        new_run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id, assistant_id=self.assistant_id
        )
        time.sleep(self.POLL_RATE)
        new_run = self.client.beta.threads.runs.retrieve(
            run_id=new_run.id, thread_id=self.thread_id
        )
        return new_run


class OpenAIChat:
    UPGRADE_MODEL = GPT4_TURBO

    def __init__(
        self,
        model,
        system_prompt,
        *,
        tools=None,
        tool_provider=None,
        format=None,
        max_tokens=1024*1024,
        temperature=0,
    ):
        self.tools = tools
        if tools:
            assert (
                tool_provider is not None
            ), "Tool provider must be given when tools are set"
        self.tool_provider = tool_provider
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.Client(api_key=api_key)

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conversation = []
        self.tokenizer = encoding_for_model(self.model)
        self.system_prompt = system_prompt
        self.format = format

        self._init_messages()

    def add_message(self, role, content):
        self.conversation.append({"role": role, "content": content})
        self.trim_conversation()

    def _init_messages(self):
        self.conversation = []
        self.add_message("system", self.system_prompt)

    def _chat_completion(self, upgrade: bool, max_retries: int = MAX_RETRIES):
        wait_time = 0.1
        model = self.model
        upgraded = False

        for i in range(max_retries):
            if upgraded:
                model = OpenAIChat.UPGRADE_MODEL
                _logger.info("Model upgraded to %s", model)

            if i > 0:
                time.sleep(wait_time)
                wait_time *= BACKOFF_FACTOR
            try:
                return self.client.chat.completions.create(
                    model=model,
                    temperature=self.temperature,
                    messages=self.conversation,
                    response_format=self.format,
                    tools=self.tools,
                )
            except openai.RateLimitError as e:
                _logger.info(f"Request rate limited {i+1} times: {e}")
            except openai.BadRequestError as e:
                if "maximum context length" in e.message and upgrade and not upgraded:
                    _logger.warn("Request too large for %s", model)
                    upgraded = True
                else:
                    raise
        raise TimeoutError("Backoff exceeded max retries")

    def handle_tool_call(self, tool):
        function_name = tool.function.name
        function_args = json.loads(tool.function.arguments)
        _logger.info("Tool call: %s(%s)", function_name, function_args)
        return self.tool_provider(function_name, function_args)

    def append_tool_call(self, tool, response):
        self.conversation.append(
            {
                "tool_call_id": tool.id,
                "role": "tool",
                "name": tool.function.name,
                "content": response,
            }
        )

    def get_response(self, message, clear=False, upgrade=False):
        self.add_message("user", message)
        _logger.info("User: %s", message)
        response = self._chat_completion(upgrade)

        message = response.choices[0].message
        _logger.info("AI: %s", message)
        self.conversation.append(message)

        while message.tool_calls:
            responses = []
            for tool in message.tool_calls:
                response = self.handle_tool_call(tool)
                responses.append(response)
                self.append_tool_call(tool, response)

            if clear:
                self._init_messages()
                return responses

            response = self._chat_completion()
            message = response.choices[0].message

        if clear:
            self._init_messages()

        return message.content

    def trim_conversation(self):
        while self.count_tokens() > self.max_tokens and len(self.conversation) > 2:
            self.conversation.pop(1)

    def count_tokens(self):
        sum = 0
        for message in self.conversation:
            if isinstance(message, dict):
                sum += len(self.tokenizer.encode(message["content"]))
            elif message.content:
                sum += len(self.tokenizer.encode(message.content))
        return sum
