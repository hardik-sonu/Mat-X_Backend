"""AFLOW AFLUX integration for computational materials records."""
from typing import Any
import math
import re
import httpx
AFLOW_ENDPOINT = "https://aflow.org/API/aflux/"
PROVIDER_NAME = "AFLOW"
DATASET_NAME = "AFLOW"
DATA_CLASSIFICATION = "COMPUTATIONAL_DFT"
VALID_ELEMENTS = set("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au
Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf
Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".split())
class AflowIntegrationError(RuntimeError):
    """Raised when AFLOW data cannot be retrieved or validated."""
def _normalize_number(value: Any, field_name: str) -> float | None:
    """Normalize a finite numeric value without confusing zero with missing."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise AflowIntegrationError(
            f"AFLOW returned an invalid boolean for {field_name}."
        )
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AflowIntegrationError(
            f"AFLOW returned a non-numeric value for {field_name}."
        ) from exc
    if not math.isfinite(number):
        raise AflowIntegrationError(
            f"AFLOW returned a non-finite value for {field_name}."
        )
    return number
def _normalize_records(
    payload: Any,
    requested_elements: list[str],
    exact_species_count: int | None,
    require_properties: bool,
) -> list[dict[str, Any]]:
    """Validate and normalize AFLOW dictionary/list response formats."""
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if any(key in payload for key in ("error", "errors", "detail")):
            raise AflowIntegrationError(
                "AFLOW returned an error payload instead of material records."
            )
        records = list(payload.values())
    else:
        raise AflowIntegrationError(
            "AFLOW returned an unsupported response structure."
        )
    normalized = []
    requested_set = set(requested_elements)
    for record in records:
        if not isinstance(record, dict):
            raise AflowIntegrationError(
                "AFLOW returned a malformed material record."
            )
        auid = record.get("auid")
        aurl = record.get("aurl")
        compound = record.get("compound")
        raw_species = record.get("species")
        if (
            not isinstance(auid, str)
            or not re.fullmatch(r"aflow:[A-Za-z0-9]+", auid)
        ):
            raise AflowIntegrationError("AFLOW record has an invalid AUID.")
        if not isinstance(aurl, str) or not aurl.strip() or ":" not in aurl:
            raise AflowIntegrationError(
                "AFLOW record has an invalid AURL locator."
            )
        if not isinstance(compound, str) or not compound.strip():
            raise AflowIntegrationError(
                "AFLOW record is missing its chemical formula."
            )
        if (
            not isinstance(raw_species, list)
            or not raw_species
            or not all(isinstance(item, str) for item in raw_species)
        ):
            raise AflowIntegrationError(
                "AFLOW record has invalid species information."
            )
        species = list(dict.fromkeys(item.strip() for item in raw_species))
        if any(item not in VALID_ELEMENTS for item in species):
            raise AflowIntegrationError(
                f"AFLOW record {auid} contains an invalid element symbol."
            )
        if not requested_set.issubset(set(species)):
            raise AflowIntegrationError(
                f"AFLOW record {auid} does not contain all requested elements."
            )
        number_of_species = len(species)
        if (
            exact_species_count is not None
            and number_of_species != exact_species_count
        ):
            raise AflowIntegrationError(
                f"AFLOW record {auid} does not match the requested "
                "exact species count."
            )
        band_gap = _normalize_number(record.get("Egap"), "Egap")
        formation_enthalpy = _normalize_number(
            record.get("enthalpy_formation_atom"),
            "enthalpy_formation_atom",
        )
        if require_properties and (
            band_gap is None or formation_enthalpy is None
        ):
            raise AflowIntegrationError(
                f"AFLOW record {auid} is missing a required property."
            )
        space_group = record.get("spacegroup_relax")
        if space_group is not None:
            if isinstance(space_group, bool):
                raise AflowIntegrationError(
                    f"AFLOW record {auid} has an invalid space group."
                )
            try:
                numeric_space_group = float(space_group)
            except (TypeError, ValueError, OverflowError) as exc:
                raise AflowIntegrationError(
                    f"AFLOW record {auid} has an invalid space group."
                ) from exc
            if (
                not math.isfinite(numeric_space_group)
                or not numeric_space_group.is_integer()
                or not 1 <= numeric_space_group <= 230
            ):
                raise AflowIntegrationError(
                    f"AFLOW record {auid} has an invalid space group."
                )
            space_group = int(numeric_space_group)
        pearson_symbol = record.get("Pearson_symbol_relax")
        if pearson_symbol is not None and not isinstance(
            pearson_symbol, str
        ):
            raise AflowIntegrationError(
                f"AFLOW record {auid} has an invalid Pearson symbol."
            )
        normalized.append({
            "auid": auid,
            "aurl": aurl,
            "compound": compound.strip(),
            "species": species,
            "number_of_species": number_of_species,
            "space_group": space_group,
            "pearson_symbol": pearson_symbol,
            "band_gap": band_gap,
            "formation_enthalpy": formation_enthalpy,
            "source": {
                "provider": PROVIDER_NAME,
                "dataset": DATASET_NAME,
                "classification": DATA_CLASSIFICATION,
                "record_id": auid,
                "record_locator": aurl,
                "data_origin": "COMPUTATIONAL",
                "citation": (
                    "Rose et al., AFLUX: The LUX materials search API "
                    "for the AFLOW data repositories, Computational "
                    "Materials Science 137 (2017), 362–370. "
                    "https://doi.org/10.1016/j.commatsci.2017.04.036"
                ),
            },
        })
    return normalized
def search_aflow_structures(
    elements: list[str],
    page: int = 1,
    page_size: int = 10,
    exact_species_count: int | None = None,
    require_properties: bool = True,
) -> dict[str, Any]:
    """Search AFLOW computational structures through AFLUX."""
    if not elements:
        raise ValueError("At least one chemical element is required.")
    normalized_elements = []
    for element in elements:
        if not isinstance(element, str):
            raise ValueError("Element symbols must be strings.")
        symbol = element.strip()
        if symbol not in VALID_ELEMENTS:
            raise ValueError(f"Invalid chemical element: {symbol}")
        if symbol not in normalized_elements:
            normalized_elements.append(symbol)
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise ValueError("page must be a positive integer.")
    if (
        not isinstance(page_size, int)
        or isinstance(page_size, bool)
        or not 1 <= page_size <= 100
    ):
        raise ValueError("page_size must be between 1 and 100.")
    if exact_species_count is not None:
        if (
            not isinstance(exact_species_count, int)
            or isinstance(exact_species_count, bool)
            or not 1 <= exact_species_count <= 7
        ):
            raise ValueError(
                "exact_species_count must be between 1 and 7."
            )
        if exact_species_count < len(normalized_elements):
            raise ValueError(
                "exact_species_count cannot be smaller than "
                "the number of requested elements."
            )
    if not isinstance(require_properties, bool):
        raise ValueError("require_properties must be a boolean.")
    matchbook = [
        f"species({','.join(normalized_elements)})"
    ]
    if exact_species_count is not None:
        matchbook.append(f"nspecies({exact_species_count})")
    if require_properties:
        matchbook.extend([
            "Egap(*)",
            "enthalpy_formation_atom(*)",
        ])
    summons = ",".join(matchbook) + (
        f",paging({page},{page_size}),format(json)"
    )
    request_url = AFLOW_ENDPOINT + "?" + summons
    try:
        response = httpx.get(request_url, timeout=60.0, follow_redirects=False)
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise AflowIntegrationError(
            "AFLOW request timed out."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise AflowIntegrationError(
            f"AFLOW returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise AflowIntegrationError(
            f"Could not connect to AFLOW: {exc}"
        ) from exc
    except ValueError as exc:
        raise AflowIntegrationError(
            "AFLOW returned invalid JSON."
        ) from exc
    records = _normalize_records(
        payload=payload,
        requested_elements=normalized_elements,
        exact_species_count=exact_species_count,
        require_properties=require_properties,
    )
    total_matches = None
    if isinstance(payload, dict) and payload:
        first_key = next(iter(payload.keys()))
        match = re.search(r"\bof\s+(\d+)\b", str(first_key))
        if match:
            total_matches = int(match.group(1))
    return {
        "status": "SUCCESS",
        "provider": PROVIDER_NAME,
        "dataset": DATASET_NAME,
        "classification": DATA_CLASSIFICATION,
        "query": {
            "elements": normalized_elements,
            "exact_species_count": exact_species_count,
            "page": page,
            "page_size": page_size,
            "require_properties": require_properties,
            "aflux_summons": summons,
        },
        "record_count": len(records),
        "total_matches": total_matches,
        "records": records,
        "property_units": {
            "band_gap": "eV",
            "formation_enthalpy": "eV/atom",
        },
        "scientific_notice": (
            "AFLOW records are computational materials data, "
            "not experimental measurements. A zero band gap is "
            "a valid calculated value and is not treated as missing. "
            "When require_properties is true, both requested properties "
            "are validated as present in every returned record. "
            "Species count is derived from the unique species list. "
            "The AFLOW AURL is a repository locator, not necessarily "
            "a directly accessible HTTP URL."
        ),
    }
