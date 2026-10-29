"""Capture the dashboard to screenshots/dashboard.png (needs playwright + chromium)."""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1180, "height": 1050})
    pg.goto("http://127.0.0.1:5000/")
    pg.wait_for_timeout(2500)
    pg.screenshot(path="screenshots/dashboard.png", full_page=True)
    b.close()
print("saved screenshots/dashboard.png")
