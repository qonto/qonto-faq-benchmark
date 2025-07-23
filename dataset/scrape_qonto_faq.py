# Download articles from support-*.qonto.com, into a set of documents:
# - `dataset/documents/metadata.json` contains `[{id, path, url, title}]`.
# - `dataset/documents/markdown/<id>_<title>.md` contains the Markdown content of the document.
# - `dataset/documents/raw/<id>_<title>.html` contains the raw content of the document.

import requests
import json
from markdownify import markdownify as md
import os
import pathlib
import re
from tqdm import tqdm
from typing import Any

DOCUMENTS_FOLDER: str = os.getenv("DOCUMENTS_FOLDER", "./dataset/documents/")
MARKETS: list[str] = ["fr", "de", "it", "es"]
HEADERS = {
    "Content-Type": "application/json",
}


def scrape_faq_articles() -> None:
    md_folder = os.path.join(DOCUMENTS_FOLDER, "markdown")
    html_folder = os.path.join(DOCUMENTS_FOLDER, "raw")
    pathlib.Path(DOCUMENTS_FOLDER).mkdir(exist_ok=True)
    pathlib.Path(md_folder).mkdir(exist_ok=True)
    pathlib.Path(html_folder).mkdir(exist_ok=True)
    for market_idx in tqdm(range(len(MARKETS)), desc=f"Fetching articles"):
        market = MARKETS[market_idx]
        for locale in [market, "en-us"]:
            start_page = get_article_start_page(market, locale)
            save_articles(market, locale, start_page)


def get_article_start_page(market: str, locale: str) -> dict[str, Any]:
    url = article_page_link(market, locale)
    return make_json_get_request(url)


def article_page_link(market: str, locale: str, limit: int = 100) -> str:
    return f"http://support-{market}.qonto.com/api/v2/help_center/{locale}/articles?sort_by=updated_at&sort_order=desc&limit={limit}"


def make_json_get_request(url: str) -> dict[str, Any]:
    response = requests.request("GET", url, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def save_articles(market: str, locale: str, current_article_page: dict[str, Any]) -> None:
    with open(os.path.join(DOCUMENTS_FOLDER, "metadata.jsonl"), "a") as meta_file:
        while current_article_page["articles"]:
            for article in current_article_page["articles"]:
                if article["draft"]:
                    continue

                # Add document metadata.

                int_id = get_id()
                id = f"{int_id:04}"
                url = article["html_url"]
                title = article["title"]
                filepath_md = os.path.join(DOCUMENTS_FOLDER, "markdown", f"{id}.md")
                filepath_html = os.path.join(DOCUMENTS_FOLDER, "raw", f"{id}.html")
                metadata = {"id": id, "path": filepath_md, "url": url, "title": title}
                json.dump(metadata, meta_file)
                meta_file.write('\n')

                # Write markdown and raw content.

                markdown = markdown_from_article(article)
                raw = article["body"]

                with open(filepath_md, "w") as md_file:
                    md_file.write(markdown)

                with open(filepath_html, "w") as html_file:
                    html_file.write(raw)

            if not current_article_page["next_page"]:
                break

            current_article_page = get_article_page_by_link(current_article_page["next_page"])


def get_article_page_by_link(link: str) -> dict[str, Any]:
    return make_json_get_request(link)


def markdown_from_article(article: dict[str, Any]) -> str:
    header = "# " + article["title"] + "\n\n"
    body = md(article["body"])
    # Remove trailing whitespace at the end of lines.
    body_cleaned = re.sub(r' +\n', '\n', body)
    # Remove extraneous newlines in paragraph splits.
    body_cleaned = re.sub(r'(\n{3,})', '\n\n', body_cleaned)
    # Remove images (syntax: ![alt](url)).
    body_cleaned = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', body_cleaned)
    return header + body_cleaned


def id_counter():
    counter = 0
    def get_id():
        nonlocal counter
        counter += 1
        return counter
    return get_id
get_id = id_counter()


if __name__ == "__main__":
    scrape_faq_articles()
