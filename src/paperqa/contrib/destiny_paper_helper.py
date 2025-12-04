import json
import logging
from pathlib import Path
from typing import Any

import anyio
import httpx
import httpx_aiohttp
from aviary.message import Message
from destiny_sdk.enhancements import EnhancementType
from destiny_sdk.identifiers import ExternalIdentifierType
from destiny_sdk.references import ReferenceSearchResult, Reference
from lmi import LiteLLMModel
from luqum.parser import parser, ParseSyntaxError
from msal import PublicClientApplication
from pydantic import Field, BaseModel, field_validator
from pydantic.v1 import ValidationError

from paperqa import Settings, Docs
from paperqa.prompts import DESTINY_search_api_docs

logger = logging.getLogger(__name__)

class DESTINYSearchQuery(BaseModel):
    query: str = Field(description="The search query to be passed to DESTINY's Search API")

class FailedToGetRelevantPapersError(Exception):
    pass

class LuceneQuery(BaseModel):
    """Validated Lucene query model."""

    query: str = Field(
        description="A valid Lucene query syntax string"
    )

    @field_validator('query')
    @classmethod
    def validate_lucene_syntax(cls, v: str) -> str:
        """Validate that the query is valid Lucene syntax."""
        try:
            parser.parse(v)
            return v
        except ParseSyntaxError as e:
            raise ValueError(f"Invalid Lucene syntax: {e}")

class DESTINYPaperHelper:
    def __init__(
            self,
            settings: Settings,
            api_url: str,
            client_id: str,
            authority: str,
            login_hint: str,
            scopes: list[str],
            search_endpoint: str = "/v1/references/search/",
            max_timeout: float = 15.0,
            max_attempts: int = 5
    ):
        self.settings = settings
        Path(settings.paper_directory).mkdir(parents=True, exist_ok=True)

        self.api_url = api_url
        self.search_endpoint = search_endpoint

        self.app = PublicClientApplication(
            client_id=client_id,
            authority=authority,
            client_credential=None
        )
        # TODO we might not need a token
        self.token = self.app.acquire_token_interactive(
            login_hint=login_hint,
            scopes=scopes
        )

        self.access_token = self.token["access_token"]

        self.max_timeout = max_timeout
        self.max_attempts = max_attempts

        self.llm_model = LiteLLMModel(
            name=self.settings.llm,
            config=self.settings.llm_config
        )
    async def fetch_relevant_papers(self, question: str)  -> dict[str, Reference]:
        """Get relevant papers/references for a given question using an LLM."""
        relevant_references = await self._get_relevant_references(question)
        await self.download_papers(relevant_references)
        return {str(ref.id):ref for ref in relevant_references}

    async def download_papers(self, references: list[Reference]) -> None:
        """Download PDFs of all relevant papers found from the DESTINY repository search."""
        downloaded_references = Path(self.settings.paper_directory).glob("*.pdf")
        downloaded_ids = {ref.stem for ref in downloaded_references}
        for ref in references:
            if str(ref.id) not in downloaded_ids:
                await self._download_pdf(ref)

    async def _download_pdf(self, reference: Reference) -> bool:
        """Download a single PDF file"""
        pdf_urls = self._parse_pdf_urls_from_reference(reference)

        async with httpx_aiohttp.HttpxAiohttpClient(
            follow_redirects=True,
            timeout=self.max_timeout
        ) as client:
            for url in pdf_urls:
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    async with await anyio.open_file(
                        f"{self.settings.paper_directory}/{str(reference.id)}.pdf", "wb"
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

    def _parse_pdf_urls_from_reference(self, ref: Reference) -> list[str]:
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

    async def _get_relevant_references(self, question: str) -> list[Reference]:
        """Perform a search using DESTINY's search API using an LLM generated search query."""
        search_query = await self._generate_lucene_search_query(question)
        for _ in range(self.max_attempts):
            try:
                resp = httpx.get(
                    f"{self.api_url}{self.search_endpoint}?q={search_query.query}",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    timeout=self.max_timeout
                )
                resp.raise_for_status()
                search_result = ReferenceSearchResult.model_validate(resp.json())
                references = search_result.references
                if not references:
                    raise ValueError(f"No references found for {search_query.query}")
                return references
            except ValidationError as e:
                print(f"Invalid response format: {e}")
                raise e
            except ValueError as e:
                print(f"Value Error: {e}")
                additional_context = f"The last search returned no references: {e}. Try creating a different search query to {search_query}."
                search_query = await self._generate_lucene_search_query(question, additional_context)
            except httpx.HTTPStatusError as e:
                print(f"HTTP Status Error: {e}")
                additional_context = f"The last search query produced: {e} with response: {resp.json()["detail"]}. Try creating a different search query to {search_query}."
                search_query = await self._generate_lucene_search_query(question, additional_context)
        raise FailedToGetRelevantPapersError(
            f"Received HTTP status errors {self.max_attempts} times. Last search_query: {search_query}"
        )

    # TODO sometimes the model generates a query that passes our validation but fails on the API call
    async def _generate_lucene_search_query(self, question: str, additional_context: str = "") -> LuceneQuery:
        prompt = f"{additional_context}\n\n" + (
            "You are the helper model that aims to generate a search query in Lucene query syntax retrieve relevant papers"
            " for the user's question from the DESTINY Repository." + "User's question:\n"
        ) + f"{question}\n\n{DESTINY_search_api_docs}"

        response = await self.llm_model.call_single(
            messages=[Message(role="user", content=prompt)],
            output_type=LuceneQuery,
            temperature=0.1
        )

        # unsure if using LuceneQuery as output type runs the validation check
        # so we explicitly run it here
        # TODO we should check if this is redundant
        lucene_query = LuceneQuery.model_validate(json.loads(str(response.text)))

        return lucene_query

    def _parse_metadata_from_reference(self, ref: Reference) -> dict[str, Any]:
        metadata = {}

        for identifier in ref.identifiers:
            if identifier.identifier_type is ExternalIdentifierType.DOI:
                metadata["doi"] = identifier.identifier

        for enhancement in ref.enhancements:
            content = enhancement.content

            match content.enhancement_type:
                case EnhancementType.BIBLIOGRAPHIC:
                    metadata["authors"] = [author.display_name for author in content.authorship]
                    metadata["title"] = content.title
                case EnhancementType.ABSTRACT:
                    metadata["abstract"] = content.abstract

        return metadata

    async def aadd_docs(
            self, references: dict[str, Reference] | None = None, docs: Docs | None = None
    ) -> Docs:
        if docs is None:
            docs = Docs()
        for doc_path in Path(self.settings.paper_directory).rglob(  # noqa: ASYNC240
                "*.pdf"
        ):
            ref = references.get(doc_path.stem) if references is not None else None
            if ref:
                metadata = self._parse_metadata_from_reference(ref)
                # TODO find a way to use bibliographic data
                try:
                    await docs.aadd(
                        doc_path,
                        settings=self.settings,
                        title=metadata.get("title", "Unknown"),
                        abstract=metadata.get("abstract", "Unknown"),
                        doi=metadata.get("doi", "Unknown"),
                        authors=metadata.get("authors", None)
                    )
                except ValueError as e:
                    logging.warning(f"Failed to aadd {doc_path} to Docs: {e}")
            else:
                await docs.aadd(doc_path, settings=self.settings)
        return docs




