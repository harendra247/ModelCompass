"""Drive the UI headlessly in demo mode and save screenshots. Usage: python scripts/ui_check.py URL OUTDIR"""
import sys, json
from playwright.sync_api import sync_playwright

url, out = sys.argv[1], sys.argv[2]
errors = []
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 900})
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.goto(url)
    pg.wait_for_selector("#models input")
    go = pg.locator("#go")
    assert go.is_disabled(), "Compare should be disabled without a prompt"
    print("reason:", pg.locator("#reason").inner_text())
    pg.fill("#prompt", "Explain Kubernetes to a developer who knows Docker")
    # AC1: <2 models disables with reason
    for v in ["mock-balanced", "mock-thorough"]:
        pg.locator(f'#models input[value="{v}"]').uncheck()
    assert go.is_disabled(); print("reason:", pg.locator("#reason").inner_text())
    for v in ["mock-balanced", "mock-thorough"]:
        pg.locator(f'#models input[value="{v}"]').check()
    pg.locator('#models input[value="mock-flaky"]').check()
    assert pg.locator("#models input:checked").count() == 4
    assert pg.locator('#models input[value="mock-hang"]').is_disabled(), "5th model must be blocked"
    pg.check("#diffs")
    go.click()
    pg.wait_for_selector(".st.running", timeout=3000)
    pg.screenshot(path=f"{out}/1-streaming.png")
    pg.wait_for_selector(".summary .md", timeout=15000)
    pg.screenshot(path=f"{out}/2-done.png", full_page=True)
    statuses = pg.locator(".card .st").all_inner_texts(); print("statuses:", statuses)
    assert statuses.count("success") == 3 and statuses.count("error") == 1
    pg.locator("text=Prefer this one").first.click()
    assert pg.locator(".card.picked").count() == 1
    pg.click("#again"); pg.wait_for_selector(".variance", timeout=15000)
    pg.wait_for_function("document.querySelectorAll('.card .st.success').length==3", timeout=15000)
    pg.wait_for_selector(".variance")
    print("tabs:", pg.locator(".tab").all_inner_texts())
    pg.screenshot(path=f"{out}/3-rerun.png", full_page=True)
    with pg.expect_download() as d: pg.click("text=Export JSON")
    data = json.load(open(d.value.path())); print("export runs:", len(data["runs"]))
    # stop mid-run
    pg.click("#go"); pg.wait_for_selector(".st.running"); pg.click("#stop")
    pg.wait_for_selector(".st.stopped"); print("stopped ok")
    pg.set_viewport_size({"width": 390, "height": 800}); pg.screenshot(path=f"{out}/4-mobile.png")
    b.close()
print("page errors:", errors)
