def should_abort_request(request):
    """Block ad/tracking/analytics traffic and heavy static assets so the
    page reaches the 'load' event quickly instead of hanging on MDL's
    continuous ad-network activity."""
    allowed_domains = ("mydramalist.com", "i.mydramalist.com")
    if not any(domain in request.url for domain in allowed_domains):
        return True

    if request.resource_type in ("image", "media", "font", "stylesheet"):
        return True

    return False