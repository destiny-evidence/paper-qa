import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from destiny_sdk.references import Reference

from paperqa import Settings
from paperqa.contrib.destiny_paper_helper import DESTINYPaperHelper, LuceneQuery
from paperqa.contrib.openalex_paper_helper import OpenAlexPaperHelper
from paperqa.settings import IndexSettings

@pytest.mark.asyncio
async def test_can_use_openalex_helper():
    with tempfile.TemporaryDirectory() as tempdir:
        settings = Settings.from_name("azure").model_copy(
            update={
                "paper_directory": tempdir,
                 "index": IndexSettings(paper_directory=tempdir )
            }
        )
        helper = OpenAlexPaperHelper(
            settings,
            email=os.getenv("OPENALEX_MAILTO"),
            api_key=os.getenv("OPENALEX_API_KEY")
        )

        question = "What is the progress on brain activity research?"

        papers = await helper.fetch_relevant_papers(question)

        docs = await helper.aadd_docs(papers)

        session = await docs.aquery(question, settings=settings)

        print(session.answer)
        assert session.answer is not None

@pytest_asyncio.fixture(scope="function")
async def helper():
    with tempfile.TemporaryDirectory() as tempdir:
        settings = Settings.from_name("azure").model_copy(
            update={
                "paper_directory": tempdir,
                "index": IndexSettings(paper_directory=tempdir )
            }
        )
        helper = DESTINYPaperHelper(
            settings,
            api_url=os.getenv("DESTINY_API_URL"),
            client_id=os.getenv("DESTINY_CLIENT_ID"),
            authority=os.getenv("DESTINY_AUTHORITY"),
            login_hint=os.getenv("DESTINY_LOGIN_HINT"),
            scopes=os.getenv("DESTINY_SCOPES").split(","),
        )
        yield helper

@pytest.mark.asyncio
async def test_can_generate_lucene_search_query_for_DESTINY_repo(helper):

    query = await helper._generate_lucene_search_query(
        "What is the progress on climate change intervention research?"
    )

    assert isinstance(query, LuceneQuery)
    assert "climate" in query.query
    assert "intervention" in query.query


@pytest.mark.asyncio
async def test_can_retrieve_relevant_papers_from_DESTINY_repo(helper):
    references = await helper._get_relevant_references(
        "What is the progress on climate change intervention research?"
    )

    assert references is not []
    assert isinstance(references[0], Reference)

@pytest.mark.asyncio
async def test_can_download_papers_from_DESTINY_sourced_references(helper):

    references = await helper._get_relevant_references(
        "What is the progress on climate change intervention research?"
    )

    await helper.download_papers(references)

    downloaded_references = Path(helper.settings.paper_directory).glob("*.pdf")
    downloaded_ids = {ref.stem for ref in downloaded_references}

    assert len(downloaded_ids) > 0
    assert downloaded_ids <= {str(ref.id) for ref in references}

@pytest.mark.asyncio
async def test_can_use_DESTINY_paper_helper(helper):
    question = "What is the progress on climate change intervention research?"

    papers = await helper.fetch_relevant_papers(question)

    docs = await helper.aadd_docs(papers)

    session = await docs.aquery(question, settings=helper.settings)

    print(session.answer)
    assert session.answer is not None