
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "raw" / "Core Rules.pdf"
OUTPUT = ROOT / "data" / "processed" / "core_rules_no_ocr.md"

try:
    if not INPUT.is_file():
        raise FileNotFoundError(f"PDF not found: {INPUT}")

    print(f"Parsing: {INPUT.name}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    print("OCR enabled: False")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(INPUT)

    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(f"Conversion status: {result.status}")

    markdown = result.document.export_to_markdown(
        page_break_placeholder="<!-- PDF_PAGE_BREAK -->"
    )

    if not markdown.strip():
        raise RuntimeError("Markdown export is empty.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(markdown, encoding="utf-8")

    print("\nParsing completed successfully.")
    print(f"PDF pages: {len(result.document.pages)}")
    print(f"Tables detected: {len(result.document.tables)}")
    print(f"Page-break markers: {markdown.count('<!-- PDF_PAGE_BREAK -->')}")
    print(f"Markdown characters: {len(markdown)}")
    print(f"Output: {OUTPUT}")

except Exception as error:
    print(f"\nParsing failed: {error}")
    raise
