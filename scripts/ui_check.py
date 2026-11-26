"""Browser smoke test of the dashboard: opens every tab in headless Chromium, reports plots/
exceptions/console errors and saves screenshots.
  streamlit run dashboard/app.py --server.headless true --server.port 8599 &
  python scripts/ui_check.py OUT_DIR    (needs: pip install playwright; playwright install chromium)
"""
import sys, time
from playwright.sync_api import sync_playwright

out = sys.argv[1]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1400, "height": 1000})
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text[:300]}"))
    pg.on("pageerror", lambda e: logs.append(f"PAGEERROR: {str(e)[:300]}"))
    for i in range(30):
        try:
            pg.goto("http://localhost:8599", timeout=5000)
            break
        except Exception:
            time.sleep(1)
    pg.wait_for_timeout(8000)
    tabs = pg.get_by_role("tab")
    print("tabs:", tabs.all_inner_texts())
    for i in range(tabs.count()):
        tabs.nth(i).click()
        pg.wait_for_timeout(3000)
        panel = pg.locator('[role="tabpanel"]:visible')
        txt = panel.first.inner_text()[:400].replace("\n", " | ") if panel.count() else "<no panel>"
        print(f"--- tab {i}: plots={pg.locator('.js-plotly-plot:visible').count()} "
              f"exceptions={pg.locator('[data-testid=stException]').count()}")
        print("   ", txt)
        pg.screenshot(path=f"{out}/tab{i}.png", full_page=True)
    print("\n".join(logs[-30:]))
    b.close()
