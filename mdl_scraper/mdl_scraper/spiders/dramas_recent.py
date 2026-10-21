import re
import scrapy
from scrapy_playwright.page import PageMethod


class DramasRecentSpider(scrapy.Spider):
    name = "dramas_recent"
    allowed_domains = ["mydramalist.com"]
    seen_urls = set()
    MAX_PAGES_PER_COUNTRY = 65  

    async def start(self):
        urls = [
            "https://mydramalist.com/search?adv=titles&ty=68&co=3&re=2021,2026&so=date",  # South Korea
            "https://mydramalist.com/search?adv=titles&ty=68&co=1&re=2021,2026&so=date",  # Japan
            "https://mydramalist.com/search?adv=titles&ty=68&co=2&re=2021,2026&so=date",  # China
            "https://mydramalist.com/search?adv=titles&ty=68&co=6&re=2021,2026&so=date",  # Thailand
        ]
        for url in urls:
            yield scrapy.Request(
                url,
                meta=dict(
                    playwright=True,
                    playwright_include_page=True,
                    playwright_page_goto_kwargs={
                        "wait_until": "domcontentloaded",
                        "timeout": 60000,
                    },
                    playwright_page_methods=[
                        PageMethod("wait_for_selector", "div.box", timeout=60000),
                    ],
                ),
                callback=self.parse_list,
            )

    async def parse_list(self, response):
        page = response.meta["playwright_page"]
        await page.close()

        links = response.css("h6.text-primary a::attr(href)").getall()
        for link in links:
            full_url = response.urljoin(link)
            if full_url in self.seen_urls:
                continue
            self.seen_urls.add(full_url)
            yield scrapy.Request(
                full_url,
                meta={
                    "playwright": True,
                    "playwright_page_goto_kwargs": {
                        "wait_until": "domcontentloaded",
                        "timeout": 60000,
                    },
                },
                callback=self.parse_drama,
            )

        if "page=" not in response.url:
            page_hrefs = response.css("a.page-link::attr(href)").getall()
            page_numbers = [
                int(m.group(1))
                for href in page_hrefs
                if (m := re.search(r'page=(\d+)', href))
            ]
            max_page = max(page_numbers) if page_numbers else 1
            max_page = min(max_page, self.MAX_PAGES_PER_COUNTRY)

            for p in range(2, max_page + 1):
                next_url = response.url + f"&page={p}"
                yield scrapy.Request(
                    next_url,
                    meta=dict(
                        playwright=True,
                        playwright_include_page=True,
                        playwright_page_goto_kwargs={
                            "wait_until": "domcontentloaded",
                            "timeout": 60000,
                        },
                        playwright_page_methods=[
                            PageMethod("wait_for_selector", "div.box", timeout=60000),
                        ],
                    ),
                    callback=self.parse_list,
                )

    # ---------- helpers ----------

    @staticmethod
    def extract_metadata_value(response, label):
        xpath_query = f'(//li[b[normalize-space(text())="{label}:"]])[1]'
        element = response.xpath(xpath_query)
        if not element:
            return None
        text_parts = element.xpath('./text() | ./*[not(self::b)]/text()').getall()
        full_text = "".join(text_parts).strip()
        full_text = re.sub(r'^[:\s#]+', '', full_text)
        return full_text if full_text else None

    @staticmethod
    def extract_tags(response):
        tag_nodes = response.xpath(
            '(//li[b[normalize-space(text())="Tags:"]])[1]'
            '//a[not(contains(@class, "edit-tags"))]/text()'
        ).getall()
        return [t.strip() for t in tag_nodes if t.strip()]

    @staticmethod
    def extract_cast(response):
        cast_list = []
        nodes = response.xpath('//ul[contains(@class, "credits")]/li')
        for node in nodes:
            actor = node.xpath('.//div[@xitempropx="actor"]//b/text()').get()
            character = node.xpath('.//div[@class="text-ellipsis"]/small/a/text()').get()
            role_type = node.xpath('.//small[@class="text-muted"]/text()').get()
            if actor:
                cast_list.append({
                    "actor": actor.strip(),
                    "character": character.strip() if character else None,
                    "role_type": role_type.strip() if role_type else None,
                })
        return cast_list

    # ---------- main parse ----------

    async def parse_drama(self, response):
        name = response.xpath('(//h1/text())[1]').get()
        if name:
            name = name.strip()

        native_title = self.extract_metadata_value(response, "Native Title")

        aka_raw = self.extract_metadata_value(response, "Also Known As")
        also_known_as = [t.strip() for t in aka_raw.split(',')] if aka_raw else []

        genres_raw = self.extract_metadata_value(response, "Genres")
        genres = [g.strip() for g in genres_raw.split(',')] if genres_raw else []

        tags = self.extract_tags(response)

        synopsis_nodes = response.xpath('//div[@class="show-synopsis"]//p//text()').getall()
        synopsis_parts = [
            t.strip() for t in synopsis_nodes
            if t.strip() and t.strip() != "Edit Translation"
        ]
        synopsis_raw = " ".join(synopsis_parts)
        synopsis = re.sub(r'\(Source:.*?\)\s*', '', synopsis_raw).strip()

        rating = self.extract_metadata_value(response, "Score")
        if rating:
            rating = rating.strip().split()[0]

        yield {
            "name": name,
            "url": response.url,
            "native_title": native_title,
            "rating": rating,
            "episodes": self.extract_metadata_value(response, "Episodes"),
            "aired_on": self.extract_metadata_value(response, "Aired On"),
            "aired_bw": self.extract_metadata_value(response, "Aired"),
            "watchers": self.extract_metadata_value(response, "Watchers"),
            "popularity": self.extract_metadata_value(response, "Popularity"),
            "original_network": self.extract_metadata_value(response, "Original Network"),
            "content_rating": self.extract_metadata_value(response, "Content Rating"),
            "country": self.extract_metadata_value(response, "Country"),
            "synopsis": synopsis,
            "genres": genres,
            "tags": tags,
            "director": self.extract_metadata_value(response, "Director"),
            "screenwriter": self.extract_metadata_value(response, "Screenwriter"),
            "also_known_as": also_known_as,
            "cast": self.extract_cast(response),
        }