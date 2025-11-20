import pytest

from paperqa import Settings, Docs, ask


@pytest.mark.asyncio
async def test_manual_controlled_agent_runs_on_azure():
    settings = Settings.from_name("azure")

    docs = Docs()

    await docs.aadd("./literature/2312.07559v2.pdf", settings=settings)

    session = await docs.aquery(
        "What is PaperQA?",
        settings=settings
    )

    assert session.answer is not None

@pytest.mark.asyncio
async def test_full_agent_runs_on_azure():
    settings = Settings.from_name("azure")

    answer_response = await ask(
        "What is PaperQA?", settings=settings
    )

    assert answer_response is not None




