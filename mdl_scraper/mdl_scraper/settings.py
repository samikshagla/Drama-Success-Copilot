DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

BOT_NAME = "mdl_scraper"
SPIDER_MODULES = ["mdl_scraper.spiders"]
NEWSPIDER_MODULE = "mdl_scraper.spiders"

PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": True,
}
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 60000
PLAYWRIGHT_ABORT_REQUEST = "mdl_scraper.utils.should_abort_request"

DOWNLOAD_DELAY = 2
RANDOMIZE_DOWNLOAD_DELAY = True
CONCURRENT_REQUESTS = 4
CONCURRENT_REQUESTS_PER_DOMAIN = 2
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 10
ROBOTSTXT_OBEY = True

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"