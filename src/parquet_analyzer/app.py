import argparse
import atexit
import pathlib
import shutil
import tempfile
import threading
import webbrowser
from functools import lru_cache

from flask import Flask, render_template, request

from ._core import (
    find_footer_segment,
    get_summary,
    parse_parquet_file,
    segment_to_json,
)
from ._html import generate_html_report, ReportElements, make_report_elements

app = Flask(__name__)

TMP_ROOT = pathlib.Path(tempfile.gettempdir()) / "parquet-analyzer"
TMP_ROOT.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def parse_parquet_file_cached(file_path: str) -> ReportElements:
    segments, _ = parse_parquet_file(file_path)
    footer = segment_to_json(find_footer_segment(segments))
    summary = get_summary(footer, segments)
    return make_report_elements(summary, footer, segments)


def generate_html(file_path, template_name):
    elements = parse_parquet_file_cached(file_path)
    section_data = [
        {
            "id": "schema",
            "title": "Schema",
        },
        {
            "id": "key-value-metadata",
            "title": "Key-value metadata",
        },
        {
            "id": "row-groups",
            "title": "Row groups",
        },
    ]
    if elements.summary.get("num_row_groups", 0) > 1:
        section_data.append(
            {
                "id": "columns",
                "title": "Columns",
            }
        )
    section_data.append(
        {
            "id": "segments",
            "title": "Segments",
        },
    )
    output = generate_html_report(
        file_path,
        elements,
        sections=[data["id"] for data in section_data],
        section_data=section_data,
        template_name=template_name,
    )
    return output


@app.route("/")
def index():
    return render_template("select.html")


@app.put("/upload")
def upload():
    app.logger.info("Receiving uploaded file...")
    tmpdir = tempfile.mkdtemp(prefix="upload_", dir=TMP_ROOT)
    file_path = pathlib.Path(tmpdir) / request.args.get("name", "uploaded.parquet")
    chunk_size = 1024 * 1024
    with open(file_path, "wb") as f:
        while True:
            chunk = request.stream.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    app.logger.info("File uploaded successfully.")
    app.logger.info("Generating report...")
    return generate_html(file_path, "viewer.html")


@app.route("/sections/<string:name>")
def section(name):
    file_path = request.args.get("file")
    return generate_html(file_path, f"sections/{name}.html")


@atexit.register
def cleanup_tmp_root():
    if TMP_ROOT.exists():
        print("Cleaning up temporary files...")
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Parquet Analyzer Web App")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    threading.Timer(
        1.0,
        lambda: webbrowser.open(f"http://{args.host}:{args.port}"),
    ).start()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
