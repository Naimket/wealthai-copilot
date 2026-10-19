"""Macro ingestion on a small synthetic PDF (the real SNB PDFs are not in the repository): no models, no network."""
import pymupdf
import pytest

from wealthai.ingestion import macro

LONG = ("The economy grew moderately in the quarter while inflation stayed low and the policy rate was unchanged. "
        "Employment rose and unemployment edged up slightly because of weaker demand abroad. ")


def make_pdf(path, with_toc=True):
    doc = pymupdf.open()
    # page 1 = cover, page 2 = body with a heading and a right-hand column
    doc.new_page().insert_text((72, 100), "Cover", fontsize=20)
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 60), "Labour market", fontname="helv", fontsize=20)
    page.insert_textbox(pymupdf.Rect(40, 100, 280, 300), LONG, fontsize=10)           # left column
    page.insert_textbox(pymupdf.Rect(320, 100, 560, 300), "Right column paragraph. " + LONG, fontsize=10)
    page.insert_text((40, 400), "2023 2024 2025", fontsize=8)                             # chart axis: must be dropped
    page.insert_text((40, 500), "Quarterly Bulletin 3 / 2026 September", fontsize=8)      # page footer: must be dropped
    if with_toc:
        doc.set_toc([[1, "Labour market", 2]])
    doc.save(path)
    return path


def test_ligature_gaps_are_repaired_but_real_words_are_kept():
    cleaned = macro.clean_text("Infl ation and signifi cant refl ection; eff ects of the aff ected sector; the staff is large")
    assert cleaned == "Inflation and significant reflection; effects of the affected sector; the staff is large"


def test_whitespace_and_hyphenation_are_normalised():
    assert macro.clean_text("a  b\nc d inter-\nnational") == "a b c d international"


def test_axis_labels_and_footers_are_dropped_and_columns_read_left_then_right(tmp_path):
    doc = macro.parse_macro_pdf(make_pdf(tmp_path / "x.pdf"), heading_styles=None)
    texts = [b.text for b in doc.blocks if b.heading_level is None]
    assert not any("2023" in t or "Quarterly Bulletin" in t for t in texts)
    assert len(texts) == 2 and texts[0].startswith("The economy") and texts[1].startswith("Right column")


def test_toc_headings_become_section_paths(tmp_path):
    chunks = macro.chunk_macro(macro.parse_macro_pdf(make_pdf(tmp_path / "x.pdf"), heading_styles=None))
    assert chunks and all(c.section == "Labour market" for c in chunks)
    assert chunks[0].text.startswith("[x | Labour market | PDF p.2]")
    assert chunks[0].isin is None and chunks[0].source == "x.pdf"


def test_document_without_bookmarks_is_one_flat_section(tmp_path):
    doc = macro.parse_macro_pdf(make_pdf(tmp_path / "flat.pdf", with_toc=False), heading_styles=None)
    chunks = macro.chunk_macro(doc)
    assert chunks and all(c.section == "flat" for c in chunks)
    assert chunks[0].text.startswith("[flat | PDF p.")  # no 'flat | flat'


def test_chunks_respect_max_size_and_never_cross_a_heading():
    blocks = [macro.MacroBlock(1, "Alpha", heading_level=1), macro.MacroBlock(1, LONG * 12),
              macro.MacroBlock(2, "Beta", heading_level=1), macro.MacroBlock(2, LONG)]
    chunks = macro.chunk_macro(macro.MacroDoc("d.pdf", "Doc", blocks))
    assert all(len(c.text) < macro.MAX_CHARS + 200 for c in chunks)
    assert {c.section for c in chunks} == {"Alpha", "Beta"}
    assert [c.section for c in chunks].index("Beta") > max(i for i, c in enumerate(chunks) if c.section == "Alpha")


def test_heading_stack_pops_to_the_right_parent():
    blocks = [macro.MacroBlock(1, "Chapter", heading_level=2), macro.MacroBlock(1, "Topic A", heading_level=3),
              macro.MacroBlock(1, LONG), macro.MacroBlock(1, "Topic B", heading_level=3), macro.MacroBlock(1, LONG)]
    sections = [c.section for c in macro.chunk_macro(macro.MacroDoc("d.pdf", "Doc", blocks))]
    assert sections == ["Chapter > Topic A", "Chapter > Topic B"]


def test_chunk_ids_are_unique_and_stable():
    blocks = [macro.MacroBlock(1, LONG * 8)]
    a = macro.chunk_macro(macro.MacroDoc("d.pdf", "Doc", blocks))
    b = macro.chunk_macro(macro.MacroDoc("d.pdf", "Doc", blocks))
    assert len({c.chunk_id for c in a}) == len(a) and [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_heading_font_filter_rejects_blocks_with_another_font(tmp_path):
    """With the SNB style list, a bookmark title typeset in a plain font (like a chart title) is not a heading."""
    doc = macro.parse_macro_pdf(make_pdf(tmp_path / "x.pdf"))  # default = SNB fonts; the synthetic heading uses Helvetica
    assert all(b.heading_level is None for b in doc.blocks)


def test_chart_data_series_are_not_prose_but_sentences_with_figures_are():
    assert not macro.is_prose("Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4 Inflation 3.2 2.1 1.6 1.6 1.2 1.4 1.1 0.7 0.4 0.0")
    assert not macro.is_prose("2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 17 18 19 20 21 22 23 24 25 26")
    assert macro.is_prose("Annual CPI inflation rose by 0.2 percentage points compared with May and stood at 0.8% in August.")
    assert not macro.is_prose("Too short to be a paragraph")


def test_glossary_entries_with_figures_and_arrows_are_prose():
    entry = ("Annualised Data is said to be annualised when it is converted to an annual value. When \u2192 GDP increases by "
             "1% from one quarter to the next, the annualised growth is 4.06%.")
    assert macro.is_prose(entry)
