import json
from pathlib import Path

import folium


BASE_DIR = Path(__file__).resolve().parent
GEOJSON_PATH = BASE_DIR / "Nep_district.geojson"
MAP_OUTPUT_PATHS = {
    "default": BASE_DIR / "static" / "nepal_map_default.html",
    "balen": BASE_DIR / "static" / "nepal_map_balen.html",
    "gagan": BASE_DIR / "static" / "nepal_map_gagan.html",
    "other": BASE_DIR / "static" / "nepal_map_kp.html",
}
INDEX_OUTPUT_PATH = BASE_DIR / "index.html"


def build_map(fill_color: str = "#4CAF50") -> folium.Map:
    with GEOJSON_PATH.open("r", encoding="utf-8") as handle:
        geojson_data = json.load(handle)

    m = folium.Map(
        location=[28.3949, 84.1240],
        zoom_start=7,
        tiles="CartoDB positron",
        control_scale=True,
    )

    def style_function(feature):
        return {
            "fillColor": fill_color,
            "color": fill_color,
            "weight": 1.2,
            "fillOpacity": 0.28,
        }

    def highlight_function(feature):
        return {
            "weight": 2.2,
            "color": "#FF8F00",
            "fillOpacity": 0.35,
        }

    folium.GeoJson(
        geojson_data,
        name="District Boundaries",
        style_function=style_function,
        highlight_function=highlight_function,
        smooth_factor=1.5,
        tooltip=folium.GeoJsonTooltip(
            fields=["DISTRICT", "PR_NAME"],
            aliases=["District", "Province"],
            localize=True,
        ),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def write_wrapper_page() -> None:
    html = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Nepal District Map</title>
    <link rel=\"stylesheet\" href=\"./static/css/style.css\">
</head>
<body>
    <div class=\"container\">
        <header>
            <h1>🗺️ Nepal District Interactive Map</h1>
            <p>District boundaries loaded from a local GeoJSON file using Folium.</p>
        </header>

        <section class=\"map-card\">
            <h2>District Boundaries</h2>
            <iframe src=\"./static/nepal_map_default.html\" title=\"Nepal District Map\" class=\"map-frame\"></iframe>
        </section>
    </div>
</body>
</html>
"""
    INDEX_OUTPUT_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    for name, output_path in MAP_OUTPUT_PATHS.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fill_color = {
            "default": "#A0A0A0",
            "balen": "#7DA1C4",
            "gagan": "#2A7E19",
            "other": "#FF0000",
        }[name]
        build_map(fill_color).save(output_path)
        print(f"Saved {name} map to {output_path}")

    write_wrapper_page()
    print(f"Saved browser page to {INDEX_OUTPUT_PATH}")
