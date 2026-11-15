"""Read-only search adapter for public Zenodo research records."""
import httpx
from typing import Any
ZENODO_API = "https://zenodo.org/api/records"
TIMEOUT_SECONDS = 20.0
USER_AGENT = "MAT-X/1.0 (materials research platform)"
class ZenodoIntegrationError(RuntimeError):
    """Raised when Zenodo returns an invalid or unavailable response."""
def _normalize_record(item: dict[str, Any]) -> dict:
    metadata = item.get("metadata") or {}
    creators = metadata.get("creators") or []
    resource_type = metadata.get("resource_type") or {}
    license_data = metadata.get("license") or {}
    links = item.get("links") or {}
    files = []
    for file_item in item.get("files") or []:
        file_links = file_item.get("links") or {}
        files.append({
            "key": file_item.get("key"),
            "size_bytes": file_item.get("size"),
            "checksum": file_item.get("checksum"),
            "download_url": file_links.get("self"),
        })
    doi = item.get("doi") or (
        (metadata.get("prereserve_doi") or {}).get("doi")
    )
    return {
        "record_id": str(item.get("id") or ""),
        "title": metadata.get("title"),
        "doi": doi,
        "publication_date": metadata.get("publication_date"),
        "description": metadata.get("description"),
        "creators": [
            {
                "name": creator.get("name"),
                "affiliation": creator.get("affiliation"),
                "orcid": creator.get("orcid"),
            }
            for creator in creators
            if isinstance(creator, dict)
        ],
        "keywords": metadata.get("keywords") or [],
        "resource_type": (
            resource_type.get("title")
            or resource_type.get("type")
        ),
        "license": license_data.get("id"),
        "record_url": links.get("html"),
        "api_url": links.get("self"),
        "files": files,
        "source_provenance": {
            "repository": "Zenodo",
            "classification": "EXTERNAL_PUBLISHED_REPOSITORY_METADATA",
            "record_id": str(item.get("id") or ""),
            "doi": doi,
        },
    }
def search_zenodo_records(
    *,
    query: str,
    page: int = 1,
    size: int = 10,
    sort: str = "bestmatch",
) -> dict:
    query = query.strip() if isinstance(query, str) else ""
    if not query:
        raise ValueError("query must not be empty.")
    if len(query) > 500:
        raise ValueError("query must not exceed 500 characters.")
    if isinstance(page, bool) or page < 1:
        raise ValueError("page must be at least 1.")
    if isinstance(size, bool) or not 1 <= size <= 25:
        raise ValueError("size must be between 1 and 25.")
    if sort not in {"bestmatch", "mostrecent", "-mostrecent"}:
        raise ValueError("Unsupported Zenodo sort order.")
    try:
        response = httpx.get(
            ZENODO_API,
            params={"q": query, "page": page, "size": size, "sort": sort},
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            timeout=TIMEOUT_SECONDS,
            follow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise ZenodoIntegrationError("Zenodo request timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise ZenodoIntegrationError(
            f"Zenodo returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise ZenodoIntegrationError("Zenodo could not be reached.") from exc
    except ValueError as exc:
        raise ZenodoIntegrationError("Zenodo returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ZenodoIntegrationError("Zenodo response must be a JSON object.")
    hits = payload.get("hits")
    if not isinstance(hits, dict):
        raise ZenodoIntegrationError("Zenodo response is missing its hits object.")
    raw_records = hits.get("hits")
    total_data = hits.get("total", 0)
    if not isinstance(raw_records, list):
        raise ZenodoIntegrationError("Zenodo response contains invalid records.")
    if isinstance(total_data, dict):
        total = total_data.get("value", 0)
    else:
        total = total_data
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ZenodoIntegrationError("Zenodo returned an invalid total count.")
    records = []
    for item in raw_records:
        if not isinstance(item, dict):
            raise ZenodoIntegrationError("Zenodo returned a malformed record.")
        records.append(_normalize_record(item))
    return {
        "status": "SUCCESS",
        "provider": "Zenodo",
        "query": query,
        "page": page,
        "page_size": size,
        "record_count": len(records),
        "total_matches": total,
        "records": records,
        "source_provenance": {
            "repository": "Zenodo",
            "classification": "EXTERNAL_PUBLISHED_REPOSITORY_METADATA",
            "api_endpoint": ZENODO_API,
        },
        "scientific_notice": (
            "This endpoint retrieves public repository metadata and file links. "
            "It does not download, validate, or scientifically certify datasets. "
            "Repository metadata and source records must be assessed before use "
            "as engineering evidence."
        ),
    }
