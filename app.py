import tempfile

import chainlit as cl
from lmi.utils import update_litellm_max_callbacks

from paperqa import agent_query, Settings
from paperqa.sources.destiny_repo import get_access_token

# Suppress LiteLLM callback warnings
update_litellm_max_callbacks()


@cl.on_chat_start
async def on_chat_start():
    print("Starting new chat session.")
    print("Attempting to retrieve DESTINY access token...")
    get_access_token()
    print("Retrieved access token successfully.")
    print("Creating paperqa settings...")
    tempdir = tempfile.TemporaryDirectory()
    cl.user_session.set("tempdir", tempdir)
    settings = Settings.from_name("search_only_destiny")
    settings.agent.index.paper_directory = tempdir.name
    settings.verbosity = 0
    cl.user_session.set("paperqa-settings", settings)


@cl.on_chat_end
def on_chat_end():
    tempdir = cl.user_session.get("tempdir")
    tempdir.cleanup()


@cl.on_message
async def main(message: cl.Message):
    # Store steps for updating them during callbacks
    current_step = None
    step_count = 0

    async def on_agent_action_callback(action, _state):
        """Called when agent takes an action (tool call)."""
        nonlocal current_step, step_count
        step_count += 1

        # Extract tool names and details from the action
        tool_names = [tc.function.name for tc in action.tool_calls]
        display_names = [name.replace("_", " ").title() for name in tool_names]
        step_name = f"{', '.join(display_names)} Tool - Step {step_count}"

        # Build detailed output for each tool call
        tool_details = []
        for tc in action.tool_calls:
            tool_name = tc.function.name

            # Check if this is a DESTINY search and extract the query
            if tool_name == "destiny_search":
                query = tc.function.arguments.get("query", "N/A")
                tool_details.append(f"**DESTINY API Search Query**: `{query}`")
                current_step = cl.Step(name=step_name, show_input=True)
            else:
                # For other tools, just show the name
                tool_details.append(f"**{tool_name}**")
                current_step = cl.Step(name=step_name, show_input=False)

        # Convert tool names to title case for display
        await current_step.__aenter__()
        current_step.input = "\n".join(tool_details)
        await current_step.update()

    async def on_env_step_callback(obs, _reward, _done, _truncated):
        """Called after environment processes the action."""
        nonlocal current_step

        if current_step is not None:
            # Format the observations (tool results)
            result_output = ""
            for msg in obs:
                if hasattr(msg, 'role') and msg.role == 'tool':
                    result_output = "\n\n" + str(msg.content)

            # Append the result to existing output instead of replacing it
            current_step.output = f"\n\n**Completed**{result_output}"
            await current_step.update()
            await current_step.__aexit__(None, None, None)
            current_step = None

    # Create a final answer step
    async with cl.Step(name="Generating Answer") as answer_step:
        answer_response = await agent_query(
            query=str(message.content),
            settings=cl.user_session.get("paperqa-settings"),
            on_agent_action_callback=on_agent_action_callback,
            on_env_step_callback=on_env_step_callback
        )
        answer_step.output = "Agent completed processing"

    await cl.Message(
        content=answer_response.session.answer
    ).send()
