import base64
import json
from typing import Any

from bs4 import (BeautifulSoup, CData, Comment, Doctype, NavigableString,
                 ResultSet, Tag)
from glob_utils import robust  # type: ignore
from glob_utils._decorators import setup_logging  # type: ignore
from pyppeteer import browser  # type: ignore

logger = setup_logging(name=__name__)


class Bs4Encoder(json.JSONEncoder):
    """
    Custom JSON encoder for BeautifulSoup objects.
    """

    def default(self, o: Any) -> Any:
        """
        Encodes BeautifulSoup objects as strings.

        Args:
                obj (Any): The object to encode.

        Returns:
                Any: The encoded object.
        """
        if isinstance(o, (BeautifulSoup)):
            return [self.default(item) for item in o.find_all()]
        if isinstance(o, (CData, Comment, Doctype, NavigableString)):
            return {o.__class__.__name__.lower(): str(o)}
        if isinstance(o, (Tag)):
            return {
                o.__class__.__name__.lower(): {
                    "attrs": dict(o.attrs),
                    "children": [self.default(item) for item in o.find_all()],
                    "text": o.text,
                }
            }
        if isinstance(o, (ResultSet)):
            return [self.default(item) for item in o]  # type: ignore
        return super().default(o)


@robust
async def get_html(page: browser.Page, *, url: str) -> str:
    """
    Fetches the HTML content of a given URL using a browser page.

    Args:
            page (browser.Page): The browser page to use for fetching the HTML.
            url (str): The URL to fetch the HTML from.

    Returns:
            str: The HTML content of the given URL.
    """
    await page.goto(url)  # type: ignore
    return await page.content()


@robust
async def get_screenshot(page: browser.Page, *, url: str) -> bytes:
    """
    Takes a screenshot of a web page.

    Args:
            page (browser.Page): The page object to use for taking the screenshot.
            url (str): The URL of the web page to capture.

    Returns:
            str: The base64-encoded string representation of the screenshot image.
    """
    await page.goto(url)  # type: ignore
    return await page.screenshot()  # type: ignore
    


@robust
async def get_json(page: browser.Page, *, url: str) -> str:
    """
    Fetches the HTML content of a given URL and returns it as a JSON string.

    Args:
            page (browser.Page): The browser page object.
            url (str): The URL to fetch the HTML content from.

    Returns:
            str: The HTML content of the URL as a JSON string.
    """
    await page.goto(url)  # type: ignore
    return json.dumps(BeautifulSoup(await page.content(), "lxml"), cls=Bs4Encoder)


@robust
async def get_pdf(page: browser.Page, *, url: str) -> str:
    """
    Retrieves a PDF from the specified URL using a browser page.

    Args:
            page (browser.Page): The browser page to use for navigation.
            url (str): The URL of the PDF to retrieve.

    Returns:
            str: The base64-encoded representation of the retrieved PDF.
    """
    await page.goto(url)  # type: ignore
    pdf = await page.pdf()  # type: ignore
    return f"data:application/pdf;base64,{base64.b64encode(pdf).decode('utf-8')}"


async def get_children(*, page: browser.Page, base_url: str):
    await page.goto(base_url)  # type: ignore

    async def get_urls(url: str) -> set[str]:
        nonlocal page
        await page.goto(url)  # type: ignore
        logger.info("Crawling %s", url)
        content = await page.content()  # type: ignore
        soup = BeautifulSoup(content, "lxml")
        urls: set[str] = set()
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and not href.startswith("#"):
                urls.add(href)  # type: ignore
        return urls

    visited_urls: set[str] = set()
    urls: set[str] = set()
    urls.update(await get_urls(url=base_url))
    while urls:
        try:
            url = urls.pop()
            if url not in visited_urls:
                yield url
                visited_urls.add(url)
                urls.update(await get_urls(url=url))
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(exc)
            continue
