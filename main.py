import json

from fastapi import FastAPI
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse,
                               Response)
from pyppeteer import browser, launch  # type: ignore
from sse_starlette import EventSourceResponse

from dom_ai_automator.ai import AIFunction, AIModel
from dom_ai_automator.automator import (get_children, get_html, get_json,
                                        get_screenshot)
from dom_ai_automator.dom import clean_object

ai = AIModel()


app = FastAPI(
	title="DOM AI Automator",
	description="A simple API for automating DOM AI tasks.",
	version="1.0.0",
)

chromium: browser.Browser # type: ignore

@app.on_event("startup")  # type: ignore
async def startup_event():
	"""
	Initializes the Chromium browser.
	"""
	global chromium # pylint: disable=W0603
	chromium = await launch(
		headless=True,
		args=["--no-sandbox"],
	)


@app.on_event("shutdown")  # type: ignore
async def shutdown_event():
	"""
	Closes the Chromium browser.
	"""
	await chromium.close()


class WebsiteFunction(AIFunction[dict[str, object]]):
	"""
	Based on the website url, returns the description of the website based on an screenshot of the website. Also returns the elements of the website on json format.
	"""

	url: str

	async def run(self) -> dict[str, object]:
		"""
		Runs the function.
		"""
		page = await chromium.newPage() 
		url = f"/api/image?url={self.url}"
		screenshot = await ai.openai_vision(
			text="This is a screenshot of a website, please comprehensively explain the details of the elements and the posible interactions and behaviors that could be tested on it as well as it's purpose and structure.",
			url=url,
		)
		json_data = await get_json(page=page, url=self.url)
		return {"screenshot": screenshot, "json": json_data}


@app.get("/api/html")  # type: ignore
async def html_response(url: str):
	"""
	Returns the HTML content of a given URL.

	Args:
					url (str): The URL to fetch the HTML from.

	Returns:
					str: The HTML content of the given URL.
	"""
	page = await chromium.newPage()
	await page.goto(url)  # type: ignore
	response = await get_html(page=page, url=url)
	return HTMLResponse(content=response)


@app.get("/api/image")  # type: ignore
async def image_response(url: str):
	"""
	Takes a screenshot of a web page.

	Args:
					url (str): The URL of the web page to capture.

	Returns:
					str: The base64-encoded string representation of the screenshot image.
	"""
	page = await chromium.newPage()
	await page.goto(url)  # type: ignore
	response = await get_screenshot(page=page, url=url)
	return Response(content=response, media_type="image/png")


@app.get("/api/json")  # type: ignore
async def json_response(url: str):
	"""
	Returns the HTML content of a given URL as a JSON string.

	Args:
					url (str): The URL to fetch the HTML content from.

	Returns:
					str: The HTML content of the URL as a JSON string.
	"""
	page = await chromium.newPage()
	await page.goto(url)  # type: ignore
	response = json.loads(await get_json(page=page, url=url))
	return JSONResponse(content=clean_object(response))


@app.get("/api/pdf")  # type: ignore
async def pdf_response(url: str):
	"""
	Returns a PDF from the specified URL.

	Args:
					url (str): The URL of the PDF to retrieve.

	Returns:
					str: The PDF file.
	"""
	page = await chromium.newPage()
	await page.goto(url)  # type: ignore
	response = await page.pdf()  # type: ignore
	return Response(content=response, media_type="application/pdf")


@app.get("/api/url", response_class=EventSourceResponse)  # type: ignore
async def children(url: str):
	"""
	Returns the children of the given URL.

	Args:
					url (str): The URL to fetch the children from.

	Returns:
					str: The children of the given URL.
	"""
	page = await chromium.newPage()
	await page.goto(url)  # type: ignore
	return EventSourceResponse(get_children(page=page, base_url=url))


@app.get("/api/vision")
async def function(text: str,url:str):
	"""
	Returns the children of the given URL.

	Args:
					url (str): The URL to fetch the children from.

	Returns:
					str: The children of the given URL.
	"""
	return await ai.openai_vision(text=text,url=url)

@app.get("/")
async def root():
	return RedirectResponse(url="/docs")

