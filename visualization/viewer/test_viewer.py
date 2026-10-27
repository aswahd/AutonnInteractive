"""Exercise the offline viewer in Chromium and compare rendered pixels to embedded voxel data."""

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright


def array(item):
    raw = np.frombuffer(gzip.decompress(base64.b64decode(item["data"])), np.uint8)
    if item["bits"]:
        raw = np.unpackbits(raw, bitorder="little")[: np.prod(item["shape"])]
    return raw.reshape(item["shape"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = dict(tested_files=[], checks=[], browser_errors=[], external_requests=[])
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        page = browser.new_page(viewport={"width": 1700, "height": 1100}, device_scale_factor=1)
        page.set_default_timeout(60000)
        page.on("pageerror", lambda error: result["browser_errors"].append(str(error)))
        page.on(
            "request",
            lambda request: (
                result["external_requests"].append(request.url) if request.url.startswith(("http:", "https:")) else None
            ),
        )
        page.context.set_offline(True)
        for file in sorted(args.output.glob("*.html")):
            if file.name == "index.html":
                continue
            text = file.read_text()
            data = json.loads(
                text.split('<script id="payload" type="application/json">', 1)[1].split("</script>", 1)[0]
            )
            page.goto(file.as_uri(), wait_until="load")
            page.wait_for_selector('body[data-ready="true"]')
            assert page.locator(".scene canvas").count() == 5, "Expected five working WebGL panels"
            assert page.locator(".slice").count() == 5
            page.screenshot(path=str(args.output / f"{file.stem}-full.png"), full_page=True)
            query = array(data["volumes"]["image"])
            for method in data["methods"]:
                page.select_option("#method", method)
                for count in ["1", "2", "3", "4"]:
                    page.select_option("#count", count)
                    expected = data["metrics"][f"{method}_k{count}"]["dice"]
                    assert f"{expected:.3f}" in page.locator('[data-kind="before"] .score').inner_text()
            for scope in ["full", "crop", "raw"]:
                page.select_option("#scope", scope)
                assert page.locator("#count").is_disabled() == (scope == "raw")
                for axis in range(3):
                    page.select_option("#axis", str(axis))
                    page.locator("#patch").evaluate("e => {e.value=e.max;e.dispatchEvent(new Event('input'));}")
                    page.locator("#slice").evaluate("e => {e.value=70;e.dispatchEvent(new Event('input'));}")
                    origin = np.zeros(3, dtype=int) if scope == "full" else np.array(data["origins"][-1])
                    point = origin.copy()
                    point[axis] += 70
                    other = [i for i in range(3) if i != axis]
                    point[other[0]] += 7
                    point[other[1]] += 5
                    pixel = page.locator('[data-kind="query"] .slice').evaluate(
                        "c => Array.from(c.getContext('2d').getImageData(5,7,1,1).data)"
                    )
                    expected = int(query[tuple(point)])
                    assert pixel == [expected, expected, expected, 255], (scope, axis, pixel, expected)
                    # Test real foreground pixels, including the raw/local versus full/global indexing.
                    current = page.input_value("#method")
                    page.locator("#opacity").evaluate("e => {e.value=1;e.dispatchEvent(new Event('input'));}")
                    for phase, color in [("before", [239, 169, 67]), ("after", [189, 156, 243])]:
                        key = (
                            f"{current}_{phase}_p{len(data['origins'])-1}"
                            if scope == "raw"
                            else f"{current}{'_autozoom' if phase == 'after' else ''}_k4"
                        )
                        mask = array(data["volumes"][key])
                        offset = np.zeros(3, dtype=int) if scope == "raw" else origin
                        cut = np.take(mask, 70 + offset[axis], axis=axis)
                        if scope != "full":
                            cut = cut[tuple(slice(int(offset[i]), int(offset[i]) + 192) for i in other)]
                        locations = np.argwhere(cut)
                        if len(locations):
                            row, col = locations[len(locations) // 2].tolist()
                            observed = page.locator(f'[data-kind="{phase}"] .slice').evaluate(
                                "(c,p) => Array.from(c.getContext('2d').getImageData(p[0],p[1],1,1).data)", [col, row]
                            )
                            assert observed == color + [255], (phase, scope, axis, observed)
                    page.locator("#opacity").evaluate("e => {e.value=.45;e.dispatchEvent(new Event('input'));}")
                page.select_option("#axis", "0")
                page.select_option("#method", data["methods"][0])
                page.locator("#patch").evaluate(
                    "(e,p) => {e.value=p;e.dispatchEvent(new Event('input'));}", data["default_patch"]
                )
                page.screenshot(path=str(args.output / f"{file.stem}-{scope}.png"), full_page=True)
            page.select_option("#support", "3")
            page.select_option("#support-overlay", "prediction")
            page.locator("#opacity").evaluate("e => {e.value=1;e.dispatchEvent(new Event('input'));}")
            page.locator("#support-slice").evaluate("e => {e.value=30;e.dispatchEvent(new Event('input'));}")
            assert "prediction" in page.locator('[data-kind="support"] .score').inner_text()
            page.click("#play")
            page.wait_for_timeout(500)
            assert page.locator("#play").inner_text() == "Pause slices"
            page.click("#play")
            page.set_viewport_size({"width": 390, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1"), "Mobile layout overflows"
            page.set_viewport_size({"width": 1700, "height": 1100})
            result["tested_files"].append(file.name)
        assert not result["browser_errors"], result["browser_errors"]
        assert not result["external_requests"], result["external_requests"]
        fallback = browser.new_page()
        fallback.add_init_script(
            "const original=HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext=function(type,...args){return type.includes('webgl')?null:original.call(this,type,...args);};"
        )
        fallback.goto((args.output / result["tested_files"][0]).as_uri())
        fallback.wait_for_selector('body[data-ready="true"]')
        assert "slice comparison is fully functional" in fallback.locator("#status").inner_text()
        result["checks"] = [
            "Offline file loading",
            "Five WebGL panels",
            "All student/count choices",
            "Full/crop/raw modes",
            "All axes and boundary window",
            "Query pixels and foreground overlays match original grid",
            "Support switching/prediction",
            "Animation",
            "Mobile layout",
            "2D fallback without WebGL",
            "No browser errors or external requests",
        ]
        browser.close()
    (args.output / "browser_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
