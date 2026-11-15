"""Validated NIST JARVIS OPTIMADE integration.
JARVIS-DFT records are computational materials data.
They must never be represented as experimental measurements.
"""
import math
from typing import Any
from urllib.parse import urljoin, urlparse
import httpx
JARVIS_ENDPOINT = (
    "https://jarvis.nist.gov/optimade/jarvisdft/v1/structures/"
)
PROVIDER_NAME = "NIST-JARVIS"
DATASET_NAME = "JARVIS-DFT"
DATA_CLASSIFICATION = "COMPUTATIONAL_DFT"
MISSING_NUMERIC_SENTINELS = {-99999, -99999.0}
ELEMENT_SYMBOLS = set("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca
Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu
Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr
Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".split())
class JarvisIntegrationError(RuntimeError):
    """Raised when JARVIS data cannot be retrieved or validated."""
def _normalize_value(value: Any) -> Any:
    """Convert known JARVIS missing-value markers to None."""
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value in MISSING_NUMERIC_SENTINELS
    ):
        return None
    return value
def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise JarvisIntegrationError(
            f"JARVIS returned an invalid {field} value."
        )
    return value
def _optional_number(value: Any, field: str) -> float | None:
    value = _normalize_value(value)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise JarvisIntegrationError(
            f"JARVIS returned an invalid numeric value for {field}."
        )
    return float(value)
def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one OPTIMADE structure."""
    if not isinstance(record, dict):
        raise JarvisIntegrationError(
            "JARVIS returned a malformed structure."
        )
    attributes = record.get("attributes")
    if not isinstance(attributes, dict):
        raise JarvisIntegrationError(
            "JARVIS structure is missing its attributes object."
        )
    jarvis_id = attributes.get("_jarvis_jid")
    if not isinstance(jarvis_id, str) or not jarvis_id.strip():
        raise JarvisIntegrationError(
            "JARVIS structure is missing its JARVIS identifier."
        )
    record_id = _optional_text(record.get("id"), "record id")
    formula = _optional_text(
        attributes.get("chemical_formula_reduced"), "formula"
    )
    space_group = _optional_text(
        attributes.get("_jarvis_spg_symbol"), "space group"
    )
    elements = attributes.get("elements")
    if elements is not None:
        if not isinstance(elements, list):
            raise JarvisIntegrationError(
                "JARVIS returned an invalid elements list."
            )
        for symbol in elements:
            if not isinstance(symbol, str) or symbol not in ELEMENT_SYMBOLS:
                raise JarvisIntegrationError(
                    "JARVIS returned an invalid chemical element symbol."
                )
    return {
        "id": record_id,
        "jarvis_id": jarvis_id.strip(),
        "formula": formula,
        "elements": elements,
        "density": _optional_number(
            attributes.get("_jarvis_density"), "density"
        ),
        "space_group": space_group,
        "formation_energy": _optional_number(
            attributes.get("_jarvis_formation_energy"),
            "formation energy",
        ),
        "band_gap": _optional_number(
            attributes.get("_jarvis_optb88vdw_bandgap"), "band gap"
        ),
        "reference": attributes.get("_jarvis_reference"),
        "source": {
            "provider": PROVIDER_NAME,
            "dataset": DATASET_NAME,
            "classification": DATA_CLASSIFICATION,
            "record_id": jarvis_id.strip(),
            "endpoint": JARVIS_ENDPOINT,
            "data_origin": "COMPUTATIONAL",
        },
    }
def search_jarvis_structures(
    elements: list[str],
    page_limit: int = 10,
) -> dict[str, Any]:
    """Search JARVIS-DFT and follow validated OPTIMADE pagination links."""
    if not isinstance(elements, list) or not elements:
        raise ValueError("At least one chemical element is required.")
    normalized_elements = []
    for element in elements:
        if not isinstance(element, str):
            raise ValueError("Element symbols must be strings.")
        symbol = element.strip()
        if symbol not in ELEMENT_SYMBOLS:
            raise ValueError(f"Invalid chemical element symbol: {symbol}")
        if symbol not in normalized_elements:
            normalized_elements.append(symbol)
    if not isinstance(page_limit, int) or isinstance(page_limit, bool):
        raise ValueError("page_limit must be an integer.")
    if not 1 <= page_limit <= 100:
        raise ValueError("page_limit must be between 1 and 100.")
    filter_expression = (
        "elements HAS ALL " + ",".join(normalized_elements)
    )
    def fetch_payload(url: str, *, initial: bool = False) -> dict[str, Any]:
        try:
            if initial:
                response = httpx.get(
                    url,
                    params={
                        "filter": filter_expression,
                        "page_limit": page_limit,
                    },
                    timeout=30.0,
                    follow_redirects=False,
                )
            else:
                response = httpx.get(
                    url,
                    timeout=30.0,
                    follow_redirects=False,
                )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise JarvisIntegrationError(
                "NIST JARVIS request timed out."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise JarvisIntegrationError(
                f"NIST JARVIS returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise JarvisIntegrationError(
                f"Could not connect to NIST JARVIS: {exc}"
            ) from exc
        except ValueError as exc:
            raise JarvisIntegrationError(
                "NIST JARVIS returned invalid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise JarvisIntegrationError(
                "NIST JARVIS response must be a JSON object."
            )
        return payload
    def parse_page(payload: dict[str, Any]) -> tuple[
        list[dict[str, Any]], str | None, bool, str
    ]:
        meta = payload.get("meta", {})
        if not isinstance(meta, dict):
            raise JarvisIntegrationError(
                "NIST JARVIS returned malformed provider metadata."
            )
        provider = meta.get("provider")
        if provider is None:
            provider_name = PROVIDER_NAME
        elif isinstance(provider, dict):
            provider_name = provider.get("name", PROVIDER_NAME)
            if (
                not isinstance(provider_name, str)
                or not provider_name.strip()
            ):
                raise JarvisIntegrationError(
                    "NIST JARVIS returned an invalid provider name."
                )
        else:
            raise JarvisIntegrationError(
                "NIST JARVIS returned malformed provider metadata."
            )
        more_data = meta.get("more_data_available")
        if more_data is not None and not isinstance(more_data, bool):
            raise JarvisIntegrationError(
                "NIST JARVIS returned invalid pagination metadata."
            )
        links = payload.get("links", {})
        if not isinstance(links, dict):
            raise JarvisIntegrationError(
                "NIST JARVIS returned malformed pagination links."
            )
        next_link = links.get("next")
        if isinstance(next_link, dict):
            next_link = next_link.get("href")
        if next_link is not None and not isinstance(next_link, str):
            raise JarvisIntegrationError(
                "NIST JARVIS returned an invalid next-page link."
            )
        records = payload.get("data")
        if not isinstance(records, list):
            raise JarvisIntegrationError(
                "NIST JARVIS response does not contain a valid data list."
            )
        normalized = [_normalize_record(record) for record in records]
        return normalized, next_link, bool(more_data), provider_name
    current_url = JARVIS_ENDPOINT
    visited_urls = set()
    normalized_records = []
    provider_name = PROVIDER_NAME
    more_data_available = False
    page_count = 0
    max_pages = 100
    while len(normalized_records) < page_limit:
        if current_url in visited_urls:
            raise JarvisIntegrationError(
                "NIST JARVIS returned a repeated pagination link."
            )
        visited_urls.add(current_url)
        payload = fetch_payload(
            current_url,
            initial=(page_count == 0),
        )
        page_records, next_link, provider_more, page_provider = (
            parse_page(payload)
        )
        if page_count == 0:
            provider_name = page_provider
        elif page_provider != provider_name:
            raise JarvisIntegrationError(
                "NIST JARVIS provider changed during pagination."
            )
        remaining = page_limit - len(normalized_records)
        normalized_records.extend(page_records[:remaining])
        provider_returned_excess = len(page_records) > remaining
        more_data_available = bool(
            provider_more or next_link or provider_returned_excess
        )
        page_count += 1
        if len(normalized_records) >= page_limit:
            break
        if not next_link:
            break
        if page_count >= max_pages:
            more_data_available = True
            break
        next_url = urljoin(current_url, next_link)
        parsed_url = urlparse(next_url)
        if (
            parsed_url.scheme != "https"
            or parsed_url.hostname != "jarvis.nist.gov"
            or parsed_url.port not in (None, 443)
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise JarvisIntegrationError(
                "NIST JARVIS returned an unsafe pagination URL."
            )
        current_url = next_url
    return {
        "status": "SUCCESS",
        "provider": provider_name,
        "dataset": DATASET_NAME,
        "classification": DATA_CLASSIFICATION,
        "query": {
            "elements": normalized_elements,
            "page_limit": page_limit,
            "filter": filter_expression,
        },
        "record_count": len(normalized_records),
        "more_data_available": more_data_available,
        "records": normalized_records,
        "scientific_notice": (
            "JARVIS-DFT values are computational results. "
            "They are not experimental measurements and must "
            "not be treated as experimental steel-property data."
        ),
    }
