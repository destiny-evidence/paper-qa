import json
import logging
from typing import Any

import anyio
import httpx
import httpx_aiohttp
from aviary.message import Message
from lmi import LiteLLMModel
from pyalex import config, Works
from pathlib import Path

from pydantic import BaseModel, Field

from paperqa import Settings, Docs

logger = logging.getLogger(__name__)


class OpenAlexSearchQuery(BaseModel):
    query: str = Field(description="The search query to passed to OpenAlex' API")


def to_oa_id(id: str) -> str:
    return id.replace("https://openalex.org/", "")


class OpenAlexPaperHelper:
    def __init__(
            self,
            settings: Settings,
            email: str | None,
            api_key: str | None
    ) -> None:
        self.settings = settings
        Path(settings.paper_directory).mkdir(parents=True, exist_ok=True)

        config.email = email
        config.api_key = api_key
        config.max_retries = 0
        config.retry_backoff_factor = 0.1
        config.retry_http_codes = [429, 500, 503]

        self.llm_model = LiteLLMModel(
            name=self.settings.llm,
            config=self.settings.llm_config,
        )


    async def fetch_relevant_papers(self, question: str) -> dict[str, Any]:
        """Get relevant papers for a given question using an LLM."""
        relevant_papers = await self._get_relevant_papers(question)
        await self.download_papers(relevant_papers)
        return {to_oa_id(paper["id"]):paper for paper in relevant_papers}


    async def _get_relevant_papers(self, question: str) -> list[dict[str, Any]]:
        """Perform a search using OpenAlex's API using an LLM generated search query."""
        prompt = (
                "You are the helper model that aims to genereate a search query to get up to 20 most relevant papers"
                " for the user's question from OpenAlex. " + "User's question:\n"
        )

        response = await self.llm_model.call_single(
            messages=[Message(role="user", content=prompt + question)],
            output_type=OpenAlexSearchQuery.model_json_schema()
        )

        query = json.loads(str(response.text))

        pages = Works().search(query).filter(has_pdf_url=True).paginate(per_page=20)
        top_20_papers = next(pages)
        return top_20_papers

    async def download_papers(self, papers: list[dict[str, Any]]) -> None:
        """Download PDFs of all papers resulting from an OpenAlex search."""
        downloaded_papers = Path(self.settings.paper_directory).glob("*.pdf")
        downloaded_ids = {p.stem for p in downloaded_papers}
        logger.info("Downloading PDFs for OpenAlex relevant papers")
        for paper in papers:
            if to_oa_id(paper["id"]) not in downloaded_ids:
                await self._download_pdf(paper)

    async def _download_pdf(self, paper: dict[str, Any]) -> bool:
        """Download a single PDF file."""
        pdf_link = paper["best_oa_location"]["pdf_url"]
        try:
            async with httpx_aiohttp.HttpxAiohttpClient(
                follow_redirects=True,
                timeout=15.0
            ) as client:
                response = await client.get(pdf_link)
                response.raise_for_status()
                async with await anyio.open_file(
                        f"{self.settings.paper_directory}/{to_oa_id(paper["id"])}.pdf", "wb"
                ) as f:
                    await f.write(response.content)
                logger.info(f"Successfully downloaded {to_oa_id(paper['id'])}.pdf")
                return True
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"Failed to download the PDF. Status code: {e.response.status_code}, text:"
                f" {response.text}"
            )
            return False
        except httpx.ReadTimeout as e:
            logger.warning(
                f"Failed to download the {to_oa_id(paper["id"])}.pdf. Timeout reached: {e}"
            )
            return False


    async def aadd_docs(
            self, papers: dict[str, Any] | None = None, docs: Docs | None = None
    ) -> Docs:
        if docs is None:
            docs = Docs()
        for doc_path in Path(self.settings.paper_directory).rglob(  # noqa: ASYNC240
                "*.pdf"
        ):
            paper = papers.get(doc_path.stem) if papers is not None else None
            if paper:
                # TODO find a way to use citations from OpenAlex
                await docs.aadd(
                    doc_path,
                    settings=self.settings,
                    title=paper["title"],
                    #abstract=paper["abstract"], # TODO consider including (not in OpenReview for eg)
                    doi=paper["doi"],
                    authors=[author["author"]["display_name"] for author in paper["authorships"]]
                )
            else:
                await docs.aadd(doc_path, settings=self.settings)
        return docs