"""Retrieve mammography-related open-access articles from PubMed Central."""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree as ET


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SEARCH_TERMS = [
    "mammography",
    "mammogram",
    "mammograms",
    "tomosynthesis",
    "DBT",
    "FFDM",
]


def create_session():
    """Create an HTTP session with retry and exponential backoff."""
    import requests
    from requests.adapters import HTTPAdapter, Retry

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def add_api_key(params: Dict[str, object]) -> Dict[str, object]:
    """Add the optional NCBI API key without exposing it on the command line."""
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def retrieve_articles(
    *,
    email: str,
    tool_name: str,
    retmax: int,
    batch_size: int,
    request_delay: float,
) -> List[Dict[str, str]]:
    """Search PMC and return article titles and body text."""
    import requests

    session = create_session()
    search_query = " OR ".join(f'"{term}"' for term in SEARCH_TERMS)
    search_params = add_api_key(
        {
            "db": "pmc",
            "term": search_query,
            "retmax": retmax,
            "retmode": "xml",
            "tool": tool_name,
            "email": email,
        }
    )

    response = session.get(ESEARCH_URL, params=search_params, timeout=30)
    response.raise_for_status()
    search_tree = ET.fromstring(response.content)
    id_list = [elem.text for elem in search_tree.findall(".//Id") if elem.text]
    print(f"Found {len(id_list)} PMC IDs for query: {search_query}")

    all_articles = []
    for start_index in range(0, len(id_list), batch_size):
        chunk_ids = id_list[start_index : start_index + batch_size]
        fetch_params = add_api_key(
            {
                "db": "pmc",
                "id": ",".join(chunk_ids),
                "retmode": "xml",
                "tool": tool_name,
                "email": email,
            }
        )

        time.sleep(request_delay)
        try:
            fetch_response = session.get(
                EFETCH_URL,
                params=fetch_params,
                timeout=60,
            )
            fetch_response.raise_for_status()
        except requests.exceptions.RequestException as error:
            print(
                f"Error fetching batch {start_index}-"
                f"{start_index + batch_size}: {error}"
            )
            continue

        batch_root = ET.fromstring(fetch_response.content)
        for article_elem in batch_root.findall("./article"):
            title_elem = article_elem.find(".//article-title")
            title = title_elem.text if title_elem is not None else "No Title"

            body_parts = []
            for elem in article_elem.findall(".//body//*"):
                if elem.tag in ["sec", "title", "p"] and elem.text:
                    body_parts.append(elem.text.strip())
            full_text = "\n".join(body_parts)

            all_articles.append({"Title": title, "Full Text": full_text})

        print(
            f"Fetched articles {start_index} to "
            f"{start_index + batch_size - 1}"
        )

    return all_articles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve mammography-related PubMed Central articles and save "
            "their title and full text to a CSV file or Excel workbook."
        )
    )
    parser.add_argument(
        "--email",
        required=True,
        help="Contact email sent to NCBI with every E-utilities request.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Destination .csv or .xlsx file.",
    )
    parser.add_argument(
        "--tool-name",
        default="MammoGRGDataBuilder",
        help="Tool identifier sent to NCBI (default: %(default)s).",
    )
    parser.add_argument(
        "--retmax",
        type=int,
        default=50000,
        help="Maximum number of PMC IDs returned by the search.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of PMC IDs fetched per request.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.3,
        help="Seconds to wait before each fetch request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd

    articles = retrieve_articles(
        email=args.email,
        tool_name=args.tool_name,
        retmax=args.retmax,
        batch_size=args.batch_size,
        request_delay=args.request_delay,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(articles, columns=["Title", "Full Text"])
    if args.output_file.suffix.lower() == ".csv":
        dataframe.to_csv(args.output_file, index=False)
    elif args.output_file.suffix.lower() in {".xlsx", ".xls"}:
        dataframe.to_excel(args.output_file, index=False)
    else:
        raise ValueError("--output-file must end in .csv, .xlsx, or .xls")
    print(f"Saved {len(dataframe)} articles to '{args.output_file}'")


if __name__ == "__main__":
    main()
