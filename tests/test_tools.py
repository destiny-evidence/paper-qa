import tempfile

import pytest
import pytest_asyncio

from paperqa import Docs, PQASession, Settings
from paperqa.agents.tools import DESTINYPaperSearch, EnvironmentState
from paperqa.settings import IndexSettings


@pytest_asyncio.fixture(scope="function")
async def azure_settings():
    with tempfile.TemporaryDirectory() as tempdir:
        settings = Settings.from_name("azure").model_copy(
            update={
                "paper_directory": tempdir,
                "index": IndexSettings(paper_directory=tempdir )
            }
        )
        yield settings

@pytest.mark.asyncio
async def test_DESTINY_search_tool(azure_settings):

    query = "climate change"

    session = PQASession(question="What is the current status on climate and health?")
    env_state = EnvironmentState(docs=Docs(), session=session)

    search_tool = DESTINYPaperSearch(
        settings=azure_settings,
    )

    result = await search_tool.destiny_search(
        query,
        env_state,
    )

    assert env_state.docs.docs, "Search did not add any papers"
    assert "Found a total of " in result


