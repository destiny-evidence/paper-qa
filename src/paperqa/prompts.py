from datetime import datetime

summary_json_prompt = (
    "Excerpt from {citation}\n\n---\n\n{text}\n\n---\n\nQuestion: {question}"
)
summary_prompt = (
    "Summarize the excerpt below to help answer a question."
    f"\n\n{summary_json_prompt}\n\n"
    "Do not directly answer the question,"
    " instead summarize to give evidence to help answer the question."
    " Stay detailed; report specific numbers, equations, or direct quotes"
    ' (marked with quotation marks). Reply "Not applicable" if the excerpt is'
    " irrelevant. At the end of your response,"
    " provide an integer score from 1-10 on a newline indicating relevance to question."  # Don't use 0-10 since we mention "not applicable" instead  # noqa: E501
    " Do not explain your score."
    "\n\nRelevant Information Summary ({summary_length}):"
)
# This prompt template integrates with `text` variable of the above `summary_prompt`
text_with_tables_prompt_template = (
    "{text}\n\n---\n\nMarkdown tables from {citation}."
    " If the markdown is poorly formatted, defer to the images."
    "\n\n{tables}"
)

# The below "cannot answer" sentinel phrase should:
# 1. Lead to complete tool being called with has_successful_answer=False
# 2. Can be used for unit testing
CANNOT_ANSWER_PHRASE = "I cannot answer"

answer_iteration_prompt_template = (
    "You are iterating on a prior answer, with a potentially different context:\n\n"
    "{prior_answer}\n\n"
    "Create a new answer only using context keys and data from the included context."
    " You can not use context keys from the prior answer which are not "
    "also included in the above context.\n\n"
)

CITATION_KEY_CONSTRAINTS = (
    "## Valid citation examples, only use comma/space delimited parentheticals:\n"
    "- (pqac-d79ef6fa, pqac-0f650d59)\n"
    "- (pqac-d79ef6fa)\n"
    "## Invalid citation examples:\n"
    "- (pqac-d79ef6fa and pqac-0f650d59)\n"
    "- (pqac-d79ef6fa;pqac-0f650d59)\n"
    "- (pqac-d79ef6fa-pqac-0f650d59)\n"
    "- pqac-d79ef6fa and pqac-0f650d59\n"
    "- Example's work (pqac-d79ef6fa)\n"
    "- (pages pqac-d79ef6fa)"
)

qa_prompt = (
    "Answer the question below with the context.\n\n"
    "Context:\n\n{context}\n\n---\n\n"
    "Question: {question}\n\n"
    "Write an answer based on the context. "
    "If the context provides insufficient information reply "
    f'"{CANNOT_ANSWER_PHRASE}." '
    "For each part of your answer, indicate which sources most support "
    "it via citation keys at the end of sentences, like {example_citation}. "
    "Only cite from the context above and only use the citation keys from the context."
    f"\n\n{CITATION_KEY_CONSTRAINTS}\n\n"
    "Do not concatenate citation keys, just use them as is. "
    "Write in the style of a scientific article, with concise sentences and "
    "coherent paragraphs. This answer will be used directly, "
    "so do not add any extraneous information.\n\n"
    "{prior_answer_prompt}"
    "Answer ({answer_length}):"
)

select_paper_prompt = (
    "Select papers that may help answer the question below. "
    "Papers are listed as $KEY: $PAPER_INFO. "
    "Return a list of keys, separated by commas. "
    'Return "None", if no papers are applicable. '
    "Choose papers that are relevant, from reputable sources, and timely "
    "(if the question requires timely information).\n\n"
    "Question: {question}\n\n"
    "Papers: {papers}\n\n"
    "Selected keys:"
)

citation_prompt = (
    "Provide the citation for the following text in MLA Format. "
    "Do not write an introductory sentence. "
    "Do not fabricate a DOI such as '10.xxxx' if one cannot be found,"
    " just leave it out of the citation. "
    f"If reporting date accessed, the current year is {datetime.now().year}\n\n"
    "{text}\n\n"
    "Citation:"
)

structured_citation_prompt = (
    "Extract the title, authors, and doi as a JSON from this MLA citation. "
    "If any field can not be found, return it as null. "
    "Use title, authors, and doi as keys, author's value should be a list of authors. "
    "{citation}\n\n"
    "Citation JSON:"
)

default_system_prompt = (
    "Answer in a direct and concise tone. "
    "Your audience is an expert, so be highly specific. "
    "If there are ambiguous terms or acronyms, first define them."
)

# NOTE: we use double curly braces here so it's not considered an f-string template
summary_json_system_prompt = (
    "Provide a summary of the relevant information"
    " that could help answer the question based on the excerpt."
    " Your summary, combined with many others,"
    " will be given to the model to generate an answer."
    " Respond with the following JSON format:"
    '\n\n{{\n  "summary": "...",\n  "relevance_score": 0-10\n}}'
    "\n\nwhere `summary` is relevant information from the text - {summary_length} words."
    " `relevance_score` is an integer 0-10 for the relevance of `summary` to the question."
    "\n\nThe excerpt may or may not contain relevant information."
    " If not, leave `summary` empty, and make `relevance_score` be 0."
)
summary_json_multimodal_system_prompt = (
    "Provide a summary of the relevant information"
    " that could help answer the question based on the excerpt."
    " Your summary, combined with many others,"
    " will be given to the model to generate an answer."
    " Respond with the following JSON format:"
    '\n\n{{\n  "summary": "...",\n  "relevance_score": 0-10,\n  "used_images": "..."\n}}'
    "\n\nwhere `summary` is relevant information from the text - {summary_length} words."
    " `relevance_score` is an integer 0-10 for the relevance of `summary` to the question."
    " `used_images` is a boolean flag indicating"
    " if any images present in a multimodal message were used,"
    " and if no images were present it should be false."
    "\n\nThe excerpt may or may not contain relevant information."
    # Don't instruct setting `used_images` to false, because if images
    # are used to determine irrelevance, then `used_images` should be true
    " If not, leave `summary` empty, and make `relevance_score` be 0."
)

env_system_prompt = (
    # Matching https://github.com/langchain-ai/langchain/blob/langchain%3D%3D0.2.3/libs/langchain/langchain/agents/openai_functions_agent/base.py#L213-L215
    "You are a helpful AI assistant."
)
env_reset_prompt = (
    "Use the tools to answer the question: {question}"
    "\n\nWhen the answer looks sufficient,"
    " you can terminate by calling the {complete_tool_name} tool."
    " If the answer does not look sufficient,"
    " and you have already tried to answer several times with different evidence,"
    " terminate by calling the {complete_tool_name} tool."
    " The current status of evidence/papers/cost is {status}"
)

# Prompt templates for use with LitQA
QA_PROMPT_TEMPLATE = "Q: {question}\n\nOptions:\n{options}"
EVAL_PROMPT_TEMPLATE = (
    "Given the following question and a proposed answer to the question, return the"
    " single-letter choice in the question that matches the proposed answer."
    " If the proposed answer is blank or an empty string,"
    " or multiple options are matched, respond with '0'."
    "\n\nQuestion: {qa_prompt}"
    "\n\nProposed Answer: {qa_answer}"
    "\n\nSingle Letter Answer:"
)

CONTEXT_OUTER_PROMPT = "{context_str}\n\nValid Keys: {valid_keys}"
EMPTY_CONTEXTS = len(CONTEXT_OUTER_PROMPT.format(context_str="", valid_keys="").strip())
CONTEXT_INNER_PROMPT_NOT_DETAILED = "{name}: {text}"
CONTEXT_INNER_PROMPT = f"{CONTEXT_INNER_PROMPT_NOT_DETAILED}\nFrom {{citation}}"

# For reference, here's Docling's image description prompt:
# https://github.com/docling-project/docling/blob/v2.55.1/docling/datamodel/pipeline_options.py#L214-L216
individual_media_enrichment_prompt_template = (
    "You are analyzing an image, formula, or table from a scientific document."
    " Provide a detailed description that will be used to answer questions about its content."
    " Focus on key elements, data, relationships, variables,"
    " and scientific insights visible in the image."
    " It's especially important to document referential information such as"
    " figure/table numbers, labels, plot colors, or legends."
    "\n\nText co-located with the media may be associated with"
    " other media or unrelated content,"
    " so do not just blindly quote referential information."
    " The smaller the image, the more likely co-located text is unrelated."
    " To restate, often the co-located text is several pages of content,"
    " so only use aspects relevant to accompanying image, formula, or table."
    "\n\nHere's a few failure modes with possible resolutions:"
    "\n- The media was a logo or icon, so the text is unrelated."
    " In this case, briefly describe the media as a logo or icon,"
    " and do not mention other unrelated surrounding text."
    "\n- The media was display type, so the text is probably unrelated."
    " The display type can be spread over several lines."
    " In this case, briefly describe the media as display type,"
    " and do not mention other unrelated surrounding text."
    "\n- The media is a margin box or design element, so the text is unrelated."
    " In this case, briefly describe the media as decorative,"
    " and do not mention other unrelated surrounding text."
    "\n- The media came from a bad PDF read, so it's garbled."
    " In this case, describe the media as garbled, state why it's considered garbled,"
    " and do not mention other unrelated surrounding text."
    "\n- The media is a subfigure or a subtable."
    " In this case, make sure to only detail the subfigure or subtable,"
    " not the entire figure or table."
    " Do not mention other unrelated surrounding text."
    "\n\n{context_text}Describe the media,"  # Allow for empty context_text
    " or if uncertain on a description please state why:"
)
full_page_enrichment_prompt_template = (
    "You are analyzing a screenshot of a page from a scientific document."
    " Provide a detailed description that will be used to answer questions about its content."
    " Focus on key elements, data, relationships, variables,"
    " and scientific insights visible in the image."
    " It's especially important to document referential information such as"
    " figure/table numbers, labels, plot colors, or legends."
    "\n\nText co-located with the screenshot may be associated with"
    " other pages' content and unrelated,"
    " so do not just blindly quote referential information."
    " To restate, the co-located text is several pages of content,"
    " so only use aspects relevant to the accompanying screenshot."
    " Do not feel the need to extensively document entities in the margins"
    " such as journal logos, display type, margin boxes, or PDF design elements."
    " If the screenshot is garbled due to a bad screenshot,"
    " describe the screenshot as garbled, state why it's considered garbled."
    "\n\n{context_text}Describe the screenshot,"  # Allow for empty context_text
    " or if uncertain on a description please state why:"
)
DESTINY_search_api_docs = """
### [API Query String Search](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id7)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#api-query-string-search "Link to this heading")

The simplest API interface for searching references is the [query string search](https://destiny-repository-prod-app.politesea-556f2857.swedencentral.azurecontainerapps.io/redoc#tag/search/operation/search_references_v1_references_search__get) at /v1/references/search/. This endpoint requires [authentication](https://destiny-evidence.github.io/destiny-repository/procedures/oauth.html).

#### [Parameters](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id8)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#parameters "Link to this heading")

The only required parameter is the query string `q`. Additional optional parameters can be provided to filter, sort, and page through results.

##### Query String (required)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#query-string-required "Link to this heading")

The `q` parameter is a query string in the [Lucene syntax](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-query-string-query.html#query-string-syntax).

At it’s simplest, this can be a simple keyword search, which will search over `title` and `abstract`:

# Get references with "climate change" anywhere in the title or abstract:
?q=climate change

# Get references with both "climate change" and "health" anywhere in the title or abstract:
?q=climate change AND health

Note

Query parameters must be [URL-encoded](https://www.w3schools.com/tags/ref_urlencode.ASP). For example, spaces must be encoded as `%20` or `+`. Most HTTP client libraries will do this automatically.

More complex queries can be constructed using the search syntax and the set of [searchable fields](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#search-fields).

# Get references with "climate", "climatology" etc in the title and either "John Doe" or "Jane Smith" as an author:
?q=title:"climat*" AND authors:("John Doe" OR "Jane Smith")

# Get references with "adaptation" or "mitigation" in the abstract that haven't yet been classified against the `Intervention` taxonomy:
?q=abstract:(adaptation OR mitigation) AND NOT evaluated_schemes:classification:taxonomy:Intervention

# Get references with "climate change" in any order and a typoed "health":
?q="change climate"~2 AND helth~

##### Start Year and End Year[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#start-year-and-end-year "Link to this heading")

The minimum and maximum publication years (inclusive) for references to return.

# Get references published from 2015 onwards:
?q=...&start_year=2015

# Get references published up to and including 2020:
?q=...&end_year=2020

# Get references published from 2015 to 2020:
?q=...&start_year=2015&end_year=2020

##### Annotations[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#annotations "Link to this heading")

The `annotation` parameter can be used to filter results based on their annotations.

These are provided in the format `<scheme>[/<label>][@score]`.

*   If an annotation is provided without a score, results will be filtered for that annotation being true

*   If a score is specified, results will be filtered for that annotation having a score greater than or equal to the given value.

*   If the label is omitted, results will be filtered if any annotation with the given scheme is true.

Multiple annotations can be provided; they will be combined using a logical `AND`.

# Get references annotated with `classification:taxonomy:Outcomes/Stroke` as true:
?q=...&annotation=classification:taxonomy:Outcomes/Stroke

# Get references with an inclusion:destiny score of at least 0.8:
?q=...&annotation=inclusion:destiny@0.8

# Get references annotated with `classification:taxonomy:Outcomes/Stroke` as true and inclusion:destiny as true:
?q=...&annotation=classification:taxonomy:Outcomes/Stroke&annotation=inclusion:destiny

##### Page[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#page "Link to this heading")

The page number of results to return. Each page is 20 results.

If omitted, defaults to the first page.

# Get the 41st to 60th results:
?q=...&page=3

##### Sort[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#sort "Link to this heading")

The field(s) to sort the results by. Use `-` prefix to sort in descending order.

If not provided, defaults to `relevance` as scored by the search engine.

Multiple sort fields can be provided; they will be applied in the order given.

# Sort by inclusion score ascending:
?q=...&sort=inclusion:destiny

# Sort by publication year descending:
?q=...&sort=-publication_year

# Sort by publication year ascending, then inclusion score descending:
?q=...&sort=publication_year&sort=-inclusion:destiny

#### [Returns](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id9)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#returns "Link to this heading")

Returns a [`ReferenceSearchResult`](https://destiny-evidence.github.io/destiny-repository/sdk/schemas.html#libs.sdk.src.destiny_sdk.references.ReferenceSearchResult "libs.sdk.src.destiny_sdk.references.ReferenceSearchResult") object.

#### [Limitations](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id10)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#limitations "Link to this heading")

There is a hard cap on the number of results at 10,000. You cannot page past this point, nor will [`total`](https://destiny-evidence.github.io/destiny-repository/sdk/schemas.html#libs.sdk.src.destiny_sdk.search.SearchResultTotal "libs.sdk.src.destiny_sdk.search.SearchResultTotal") show more than this.

### [API Lookup](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id11)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#api-lookup "Link to this heading")

Though not strictly a search, the [lookup endpoint](https://destiny-repository-prod-app.politesea-556f2857.swedencentral.azurecontainerapps.io/redoc#tag/v1/operation/lookup_references_v1_references__get) at /v1/references/ can be used to retrieve references by their identifiers. This endpoint requires [authentication](https://destiny-evidence.github.io/destiny-repository/procedures/oauth.html).

#### [Parameters](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id12)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id1 "Link to this heading")

##### Identifiers (required)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#identifiers-required "Link to this heading")

The identifier(s) to look up. Multiple identifiers can be provided, either in a comma-separated list or as multiple parameters.

Identifiers are in the format `[[<other>:]<type>:]<identifier>`:

*   If looking up a reference by its Destiny UUID4 id, no type prefix is needed: `09547790-7dfe-455e-a8df-5dca91963a5b`.

*   If looking up a reference by a supported [`identifier type`](https://destiny-evidence.github.io/destiny-repository/sdk/schemas.html#libs.sdk.src.destiny_sdk.identifiers.ExternalIdentifierType "libs.sdk.src.destiny_sdk.identifiers.ExternalIdentifierType"), the type must be prefixed: `doi:10.1000/xyz123`.

*   If looking up a reference by a custom identifier type, the type must be prefixed with `other:`: `other:custom:internal-id-001`.

#### [Returns](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id13)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id2 "Link to this heading")

Returns a list of [`Reference`](https://destiny-evidence.github.io/destiny-repository/sdk/schemas.html#libs.sdk.src.destiny_sdk.references.Reference "libs.sdk.src.destiny_sdk.references.Reference") objects in [deduplicated](https://destiny-evidence.github.io/destiny-repository/procedures/deduplication.html#deduplicated-projection) form.

#### [Limitations](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id14)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id3 "Link to this heading")

There is a hard cap of 100 identifiers per request. If more are needed, multiple requests must be made.

[Search Fields](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id15)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#search-fields "Link to this heading")
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### [Search Field Selection](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id16)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#search-field-selection "Link to this heading")

References may have multiple sources of information, so search fields are collapsed into a single set of searchable fields. The relevant data is prioritised by:

*   Fields provided on the [canonical reference](https://destiny-evidence.github.io/destiny-repository/procedures/deduplication.html) are prioritised over those on duplicate references.

*   Then, the most recently added data is prioritised.

### [Bibliographic](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id17)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#bibliographic "Link to this heading")

ReferenceSearchFieldsMixin.title _str_[[source]](https://github.com/destiny-evidence/destiny-repository/blob/main/app/domain/references/models/es.py)
The title of the reference.

ReferenceSearchFieldsMixin.authors _list[str]_[[source]](https://github.com/destiny-evidence/destiny-repository/blob/main/app/domain/references/models/es.py)
The authors of the reference.

These are ordered by:

*   First author

*   Middle authors in alphabetical order

*   Last author

ReferenceSearchFieldsMixin.publication_year _int_[[source]](https://github.com/destiny-evidence/destiny-repository/blob/main/app/domain/references/models/es.py)
The publication year of the reference.

### [Abstract](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id18)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#abstract "Link to this heading")

ReferenceSearchFieldsMixin.abstract _str_[[source]](https://github.com/destiny-evidence/destiny-repository/blob/main/app/domain/references/models/es.py)
The abstract of the reference.

### [Annotations](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id19)[#](https://destiny-evidence.github.io/destiny-repository/procedures/search.html#id5 "Link to this heading")

ReferenceSearchFieldsMixin.annotations _list[str]_[[source]](https://github.com/destiny-evidence/destiny-repository/blob/main/app/domain/references/models/es.py)
Every `true` annotation on the reference.

These are in format `<scheme>[/<label>]`.

Examples:

*   `classification:taxonomy:Outcomes/Stroke`

*   `classification:taxonomy:Intervention/Climate policy instruments`

*   `inclusion:destiny` (No label)

ReferenceSearchFieldsMixin.evaluated_schemes _list[str]_[[source]](https://github.com/destiny-evidence/destiny-repository/blob/main/app/domain/references/models/es.py)
Every scheme that has been evaluated for this reference.

Combining this with `annotations` allows you to determine which annotations were evaluated as `false`.

Examples:

*   `inclusion:destiny`

*   `classifier:taxonomy:Outcomes`

ReferenceSearchFieldsMixin.inclusion_destiny _float[0-1]_[[source]](https://github.com/destiny-evidence/destiny-repository/blob/main/app/domain/references/models/es.py)
The destiny inclusion score for this reference.

This is used to apply custom thresholds for inclusion. If you just want to know if the reference was included per the default threshold, check for `inclusion:destiny` in `annotations`.
"""