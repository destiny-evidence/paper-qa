import logging
import os
from functools import lru_cache
from http import HTTPStatus
from pathlib import Path
from typing import Any

import anyio
import httpx
import httpx_aiohttp
from destiny_sdk.enhancements import EnhancementType
from destiny_sdk.identifiers import ExternalIdentifierType
from destiny_sdk.references import Reference, ReferenceSearchResult
from msal import PublicClientApplication
from pydantic import ValidationError

from paperqa.docs import Docs
from paperqa.settings import Settings

logger = logging.getLogger(__name__)

API_URL = os.getenv("DESTINY_API_URL")
CLIENT_ID = os.getenv("DESTINY_CLIENT_ID")
AUTHORITY = os.getenv("DESTINY_AUTHORITY")
LOGIN_HINT = os.getenv("DESTINY_LOGIN_HINT")
SCOPES = os.getenv("DESTINY_SCOPES").split(",")
SEARCH_ENDPOINT = "/v1/references/search/"

class InvalidDESTINYQuerySyntaxError(Exception):
    pass

class DESTINYSearchPageIndexError(Exception):
    pass

class NoDESTINYResultsFoundError(Exception):
    pass

class NoMoreDESTINYSearchResultsAvailable(Exception):
    pass

class DESTINYSearchAPIError(Exception):
    pass

class DESTINYSearchAPIClientError(Exception):
    pass

class MiscDESTINYSearchAPIError(Exception):
    pass

@lru_cache(maxsize=1)
def get_access_token():
    app = PublicClientApplication(
        client_id=CLIENT_ID,
        authority=AUTHORITY,
        client_credential=None
    )
    token = app.acquire_token_interactive(
        login_hint=LOGIN_HINT,
        scopes=SCOPES,
    )
    return token["access_token"]


async def add_destiny_references_to_docs(
        query: str,
        docs: Docs,
        settings: Settings,
        page: int,
        max_timeout: float = 30.0
) -> tuple[int, int, str | None]:
    initial_docs_size = len(docs.texts)
    total_result_count = 0
    try:
        access_token = get_access_token()
        references, total_result_count = await get_relevant_references(query, access_token, page, max_timeout)
        await download_papers(references, settings.paper_directory, max_timeout)
        assert references is not None, "references is None for some reason"
        references = {str(ref.id):ref for ref in references}
        await aadd_docs(references, docs, settings)
        return total_result_count, len(docs.texts) - initial_docs_size, None
    except Exception as e:
        return total_result_count, len(docs.texts) - initial_docs_size, str(e)

async def download_papers(references: list[Reference], paper_directory: str | os.PathLike, max_timeout: float) -> None:
    """Download PDFs of all relevant papers found from the DESTINY repository search."""
    downloaded_references = Path(paper_directory).glob("*.pdf")
    downloaded_ids = {ref.stem for ref in downloaded_references}
    for ref in references:
        if str(ref.id) not in downloaded_ids:
            await download_pdf(ref, paper_directory, max_timeout)

async def download_pdf(
        reference: Reference,
        paper_directory: str | os.PathLike,
        max_timeout: float
) -> bool:
    """Download a single PDF file"""
    pdf_urls = parse_pdf_urls_from_reference(reference)
    async with httpx_aiohttp.HttpxAiohttpClient(
            follow_redirects=True,
            timeout=max_timeout
    ) as client:
        for url in pdf_urls:
            try:
                response = await client.get(url)
                response.raise_for_status()
                doc_path = f"{paper_directory}/{str(reference.id)}.pdf"
                async with await anyio.open_file(
                        doc_path, "wb"
                ) as f:
                    await f.write(response.content)
                logger.info(f"Successfully downloaded {str(reference.id)}.pdf")
                return True
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"Failed to download the PDF. Status code: {e.response.status_code}, text:" 
                    f" {response.text}"
                )
            except httpx.ReadTimeout as e:
                logger.warning(
                    f"Failed to download the {str(reference.id)}.pdf. Timeout reached: {e}"
                )
        return False

async def aadd_docs(
        references: dict[str, Reference],
        docs: Docs,
        settings: Settings
) -> None:
    # TODO we probably need to rethink this approach, currently we're looping through all downloaded files
    # and attempt to add their pdf to the Docs. But we're revisiting previously downloaded + added pdfs here
    for doc_path in Path(settings.paper_directory).rglob("*.pdf"):
        ref = references.get(doc_path.stem) if references is not None else None
        if ref:
            metadata = parse_metadata_from_reference(ref)
            await docs.aadd(
                doc_path,
                settings=settings,
                title=metadata.get("title", "Unknown"),
                abstract=metadata.get("abstract", "Unknown"),
                doi=metadata.get("doi", "Unknown"),
                authors=metadata.get("authors", None)
            )

def parse_metadata_from_reference(ref: Reference) -> dict[str, Any]:
    metadata = {}

    assert ref.identifiers is not None, "A reference's identifiers is None"
    assert ref.enhancements is not None, "A reference's enhancements is None"


    for identifier in ref.identifiers:
        if identifier.identifier_type is ExternalIdentifierType.DOI:
            metadata["doi"] = identifier.identifier

    for enhancement in ref.enhancements:
        content = enhancement.content
        match content.enhancement_type:
            case EnhancementType.BIBLIOGRAPHIC:
                if content.authorship:
                    metadata["authors"] = [author.display_name for author in content.authorship]
                metadata["title"] = content.title
            case EnhancementType.ABSTRACT:
                metadata["abstract"] = content.abstract

    return metadata


def parse_pdf_urls_from_reference(ref: Reference) -> list[str]:
    pdf_urls = []
    for enhancement in ref.enhancements:
        metadata = enhancement.content
        if metadata.enhancement_type is EnhancementType.LOCATION:
            # pdf urls are instances of HttpUrl so need to be cast to strings
            pdf_urls += [
                str(location.pdf_url) for location in metadata.locations
                if location.pdf_url is not None
            ]
    return pdf_urls


async def get_relevant_references(
        query: str,
        access_token: str,
        page: int,
        max_timeout: float
) -> tuple[list[Reference], int]:
    """Use the DESTINY repo's Search API to retrieve references for a given search query on a given page"""
    if not query.startswith("?q="):
        query = f"?q={query}"

    try:
        resp = httpx.get(
            f"{API_URL}{SEARCH_ENDPOINT}{query}&page={page}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=max_timeout
        )
        resp.raise_for_status()
        search_result = ReferenceSearchResult.model_validate(resp.json())
        n_available_references = search_result.total.count
        if n_available_references == 0:
            raise NoDESTINYResultsFoundError(f"No references found for {query}")
        # if the lower bound of the given page range is greater than the
        # total available references, then all future pages will be empty
        if (20 * (page-1))+1 > n_available_references:
            raise NoMoreDESTINYSearchResultsAvailable(
                f"There are only {n_available_references} references available. " 
                f"This page {page} and all future pages will be empty."
            )
        references = search_result.references
        return references, n_available_references
    except ValidationError as e:
        raise DESTINYSearchAPIError(f"API didn't return a valid ReferenceSearchResult object: {e}")
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        content = e.response.content
        match status:
            case HTTPStatus.UNPROCESSABLE_ENTITY:
                raise DESTINYSearchPageIndexError(
                    f"Page number {page} for {query} must not exceed 500. "
                    "Each search is limited to 10000 results with page size 20."
                )
            case HTTPStatus.BAD_REQUEST:
                raise InvalidDESTINYQuerySyntaxError(
                    f"API failed to parse search {query}, received: {content}."
                )
            case _:
                raise MiscDESTINYSearchAPIError(
                    f"DESTINY search api failed with status {status} and exception {e}."
                )
    except (NoMoreDESTINYSearchResultsAvailable, NoDESTINYResultsFoundError) as e:
        raise e
    except Exception as e:
        raise MiscDESTINYSearchAPIError(
            f"DESTINY search API failed: {e}."
        )





