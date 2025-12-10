import tempfile

import pytest
import pytest_asyncio

from paperqa.settings import Settings, IndexSettings
from paperqa.sources.destiny_repo import get_access_token, get_relevant_references, DESTINYSearchPageIndexError, \
    NoMoreDESTINYSearchResultsAvailable, NoDESTINYResultsFoundError, InvalidDESTINYQuerySyntaxError


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

@pytest_asyncio.fixture(scope="module")
async def destiny_access_token():
    return get_access_token()

@pytest.mark.asyncio
async def test_can_get_relevant_references_from_DESTINY_repo(destiny_access_token):

    query = "?q=climate change AND health"
    references = await get_relevant_references(
        query,
        destiny_access_token,
        1,
        30.0
    )

    assert len(references) > 0

@pytest.mark.asyncio
async def test_retrieving_DESTINY_references_with_out_of_range_index_fails(destiny_access_token):
    query = "?q=climate change AND health"
    with pytest.raises(DESTINYSearchPageIndexError) as exc_info:
        await get_relevant_references(
            query,
            destiny_access_token,
            5000,
            30.0
        )
        assert exc_info is not None

@pytest.mark.asyncio
async def test_retrieving_DESTINY_references_with_too_large_page_fails(destiny_access_token):
    query = "?q=climate change AND lion"
    with pytest.raises(NoMoreDESTINYSearchResultsAvailable) as exc_info:
        await get_relevant_references(
            query,
            destiny_access_token,
            250,
            15.0
        )
        assert exc_info is not None

@pytest.mark.asyncio
async def test_retrieving_DESTINY_references_with_no_search_results_fails(destiny_access_token):
    query = "?q=capybara AND polymorphism"
    with pytest.raises(NoDESTINYResultsFoundError) as exc_info:
        await get_relevant_references(
            query,
            destiny_access_token,
            1,
            15.0
        )
        assert exc_info is not None


@pytest.mark.asyncio
async def test_cant_retrieve_DESTINY_references_with_invalid_search_query_syntax(destiny_access_token):
    query = "?q=abstract:(adaptation OR mitigation) AND NOT evaluated_schemes:classification:taxonomy:Intervention"
    with pytest.raises(InvalidDESTINYQuerySyntaxError) as exc_info:
        await get_relevant_references(
            query,
            destiny_access_token,
            1,
            30.0
        )
        assert exc_info is not None