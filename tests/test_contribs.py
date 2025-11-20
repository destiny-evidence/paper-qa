import os
import tempfile

import pytest

from paperqa import Settings
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
