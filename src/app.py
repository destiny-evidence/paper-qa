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
    answer_response = await agent_query(
        query=str(message.content),
        settings=cl.user_session.get("paperqa-settings")
    )

    await cl.Message(
        content=answer_response.session.answer
    ).send()
