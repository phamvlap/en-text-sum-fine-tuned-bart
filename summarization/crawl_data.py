import re
import requests
import pandas as pd
import time

from pathlib import Path
from typing import Optional, Any
from bs4 import BeautifulSoup, Tag
from requests.exceptions import (
    Timeout,
    TooManyRedirects,
    RequestException,
    HTTPError,
    ConnectionError,
)

BASE_URL_PATTERN = r"^(http[s]?://[^/]+)"
LANG = "en"


def get_html(url: str) -> Optional[str]:
    try:
        response = requests.get(url)

        if response.status_code != 200:
            return None

        return response.text
    except (Timeout, TooManyRedirects, ConnectionError, RequestException, HTTPError):
        return None


def get_base_url(url: str) -> Optional[str]:
    base_url_pattern = re.compile(pattern=BASE_URL_PATTERN, flags=re.IGNORECASE)

    base_url_groups = base_url_pattern.match(url)
    base_url = None

    if base_url_groups is not None:
        base_url = base_url_groups.group(0)

    return base_url


def get_urls_from_html_page(url: str) -> list[str]:
    base_url = get_base_url(url=url)
    if base_url is None:
        return []

    html = get_html(url=url)
    if html is None:
        return []

    soup = BeautifulSoup(html, "html.parser")
    a_html_tags = soup.find_all("a")

    postfix_urls: list[str] = []
    for a_tag in a_html_tags:
        href_value = a_tag["href"].strip() if a_tag.has_attr("href") else None
        if (
            href_value is not None
            and href_value.startswith(f"/{LANG}/")
            and href_value not in postfix_urls
            and "tag" not in href_value
        ):
            postfix_urls.append(href_value)

    urls = [base_url + postfix_url for postfix_url in postfix_urls]

    return urls


def is_html_group(url: str) -> bool:
    return (not url.endswith(".html")) or (url.endswith(".html") and "tag" in url)


def crawl_data(url: str, total: int, verbose: bool = True) -> list[dict[str, str]]:
    base_url = get_base_url(url=url)

    if base_url is None:
        raise ValueError(f"Not support base URL from {url}")

    visited_urls: list[str] = []
    url_list: list[str] = []
    data: list[dict[str, str]] = []
    count = 0

    if is_html_group(url=url):
        url_list = get_urls_from_html_page(url=url)
    else:
        url_list = [url]

    while len(url_list) > 0 and count < total:
        current_url = url_list.pop(0)
        visited_urls.append(current_url)

        if verbose:
            print(f"[{count + 1}/{total}] Processing {current_url}...")

        html = get_html(url=current_url)
        if html is None:
            continue

        soup = BeautifulSoup(html, "html.parser")

        if "tag" not in current_url:
            # get title
            title = ""
            title_tag = soup.title
            if title_tag is not None:
                title = title_tag.string.strip() if title_tag.string is not None else ""

            # get main contenta tag
            main_content_tag = soup.find("div", id="maincontent")

            # get image captions
            image_tags = []
            if main_content_tag is not None and isinstance(main_content_tag, Tag):
                image_tags = main_content_tag.find_all("table", class_="image")
                image_tags += main_content_tag.find_all("figure")

            image_captions: list[str] = []
            for image in image_tags:
                caption = image.find("p")
                if caption is not None:
                    image_captions.append(caption.get_text().strip())
                caption = image.find("figcaption")
                if caption is not None:
                    image_captions.append(caption.get_text().strip())

            # get inner articles
            inner_article_tag = None
            if main_content_tag is not None and isinstance(main_content_tag, Tag):
                inner_article_tag = main_content_tag.find("div", class_="inner-article")

            inner_articles: list[str] = []
            if inner_article_tag is not None and isinstance(inner_article_tag, Tag):
                p_tags = inner_article_tag.find_all("p")
                for p_tag in p_tags:
                    inner_articles.append(p_tag.get_text().strip())

            # get control texts
            control_text_tags = None
            if main_content_tag is not None and isinstance(main_content_tag, Tag):
                control_text_tags = main_content_tag.find_all(
                    name="p",
                    class_="vjs-control-text",
                )
            control_texts: list[str] = []
            if control_text_tags is not None:
                for control_text in control_text_tags:
                    control_texts.append(control_text.get_text().strip())

            # get paragraphs
            paragraph_tags = []
            if main_content_tag is not None and isinstance(main_content_tag, Tag):
                paragraph_tags = main_content_tag.find_all(["p", "figure"])

            cleaned_paragraphs = []
            for _, p_tag in enumerate(paragraph_tags):
                if p_tag.find("picture"):
                    continue
                text = p_tag.get_text().strip()
                if (
                    text != ""
                    and text not in image_captions
                    and text not in inner_articles
                    and text not in control_texts
                ):
                    cleaned_paragraphs.append(p_tag)

            # get content
            content = ""
            for _, p_tag in enumerate(cleaned_paragraphs):
                text = p_tag.get_text().strip()
                if re.match(r"^[\.\'\")}\]]$", text[-1]):
                    content += text + "\n"
                else:
                    last_dot_char_position = text.rfind(".")
                    if last_dot_char_position != -1:
                        content += text[: last_dot_char_position + 1].strip() + "\n"
            content = content.strip()

            # get summary
            summary = ""
            contained_summary_tag = soup.find("h2", class_="content-detail-sapo")
            if contained_summary_tag is not None and isinstance(
                contained_summary_tag, Tag
            ):
                summary = contained_summary_tag.get_text().strip()

            # add new data
            news = {
                "url": current_url,
                "title": title,
                "content": content,
                "summary": summary,
            }
            data.append(news)
            count += 1

            if count >= total:
                break

        all_a_tags = soup.find_all("a")
        related_urls: list[str] = []

        for a_tag in all_a_tags:
            href_value = a_tag["href"] if a_tag.has_attr("href") else None
            if (
                href_value is not None
                and href_value.startswith(f"/{LANG}/")
                and href_value not in related_urls
            ):
                href_value = re.sub("#.*$", "", href_value).strip()
                related_urls.append(href_value)

        related_urls = [base_url + url for url in related_urls]
        for url in related_urls:
            if url not in visited_urls and url not in url_list:
                url_list.append(url)

    return data


def save_data(data: list[dict[str, str]], filepath: str | Path) -> None:
    df = pd.DataFrame(data)

    filepath = str(filepath)
    path_splits = filepath.rsplit("/", 1)
    dir_path = None
    if len(path_splits) > 1:
        dir_path = path_splits[0]
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    new_filepath = filepath
    if not filepath.endswith(".csv"):
        new_filepath = f"{dir_path}/data.csv" if dir_path is not None else "data.csv"
        print(
            f"File path {filepath} is not a CSV file. Save to CSV file {new_filepath} instead."
        )
    df.to_csv(new_filepath, index=False)

    print(f"Data saved to {new_filepath}")
    print(f"Size of data: {df.shape}")
    print(f"Columns: {list(df.columns)}")


def main(config: dict[str, Any]):
    print("Crawling data...")
    start = time.time()

    url = config["url"]
    total = config["total"] if config["total"] > 0 else 0

    if total == 0:
        print(f"Invalid total value: {config['total']}")
        return

    data = crawl_data(url=url, total=total, verbose=config["verbose"])

    end = time.time()

    if len(data) == 0:
        print("No data crawled")
        return

    output_file_path = config["output_path"]
    save_data(data=data, filepath=output_file_path)

    print("Done!")
    print(f"Time elapsed: {end - start:.4f} seconds")
    print(f"Speed: {total / (end - start):.2f} articles/second")
