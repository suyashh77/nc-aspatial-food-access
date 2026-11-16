"""Convert the handover markdown docs to styled Word files.
Handles: # headings, paragraphs, - / 1. lists (2 levels), - [ ] checkboxes, > quotes,
| tables |, ``` code blocks, **bold**, *italic*, `code`, [text](url) links (as text).
usage: python md2docx.py in.md out.docx
"""
import re
import sys
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

FONT, MONO = "Calibri", "Consolas"
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x5F, 0x6B)
HEADER_FILL, CODE_FILL, QUOTE_FILL = "DCE6F0", "F2F2F2", "EEF3F8"
INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*|\[[^\]]+\]\([^)]+\))")


def shade(el, fill):
    pr = el.get_or_add_tcPr() if el.tag.endswith("}tc") else el.get_or_add_pPr()
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), fill)
    pr.append(s)


def add_runs(p, text, size=None, bold=False):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`"):
            r = p.add_run(part[1:-1]); r.font.name = MONO; r.font.color.rgb = NAVY
        elif part.startswith("[") and "](" in part:
            r = p.add_run(part[1:part.index("](")])
            r.font.color.rgb = NAVY; r.underline = True
        elif part.startswith("*") and len(part) > 2:
            r = p.add_run(part[1:-1]); r.italic = True
        else:
            r = p.add_run(part)
        if bold:
            r.bold = True
        if size:
            r.font.size = Pt(size)


def setup(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(1))
    st = doc.styles
    n = st["Normal"]; n.font.name = FONT; n.font.size = Pt(11)
    n.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    n.paragraph_format.space_after = Pt(6); n.paragraph_format.line_spacing = 1.15
    for name, size, before, color in (("Title", 24, 0, NAVY), ("Heading 1", 16, 18, NAVY),
                                      ("Heading 2", 13, 14, NAVY), ("Heading 3", 11.5, 10, GREY)):
        s = st[name]; s.font.name = FONT; s.font.size = Pt(size); s.font.bold = True
        s.font.color.rgb = color
        rf = s.element.rPr.rFonts
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(a), FONT)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]
        pf = s.paragraph_format; pf.space_before = Pt(before); pf.space_after = Pt(6); pf.keep_with_next = True
    # footer page number
    fp = sec.footer.paragraphs[0]; fp.alignment = 1
    r = fp.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
        r._r.append(e)
    r.font.size = Pt(9); r.font.color.rgb = GREY


def table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    ncol = max(len(r) for r in cells)
    t = doc.add_table(rows=len(cells), cols=ncol)
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, r in enumerate(cells):
        for j in range(ncol):
            c = t.cell(i, j); p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(2); p.paragraph_format.line_spacing = 1.0
            add_runs(p, r[j] if j < len(r) else "", size=9.5, bold=(i == 0))
            if i == 0:
                shade(c._tc, HEADER_FILL)
    # repeat header row on page breaks
    trPr = t.rows[0]._tr.get_or_add_trPr(); h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trPr.append(h)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def convert(src, dst):
    doc = Document(); setup(doc)
    with open(src, encoding="utf-8") as f:
        lines = f.read().splitlines()
    i, first_h1 = 0, True
    while i < len(lines):
        ln = lines[i]; s = ln.strip()
        if s.startswith("```"):
            i += 1; code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            p = doc.add_paragraph(); shade(p._p, CODE_FILL)
            p.paragraph_format.left_indent = Inches(0.15); p.paragraph_format.line_spacing = 1.0
            for k, c in enumerate(code):
                r = p.add_run(c); r.font.name = MONO; r.font.size = Pt(8.5)
                if k < len(code) - 1:
                    r.add_break()
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i]); i += 1
            table(doc, rows); continue
        elif m := re.match(r"^(#{1,4})\s+(.*)", s):
            lvl = len(m.group(1))
            if lvl == 1 and first_h1:
                p = doc.add_paragraph(style="Title"); add_runs(p, m.group(2)); first_h1 = False
            else:
                p = doc.add_heading(level=min(lvl if lvl > 1 else 1, 3)); add_runs(p, m.group(2))
        elif m := re.match(r"^(\s*)[-*]\s+\[( |x)\]\s+(.*)", ln):
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, ("☑ " if m.group(2) == "x" else "☐ ") + m.group(3))
        elif m := re.match(r"^(\s*)[-*]\s+(.*)", ln):
            p = doc.add_paragraph(style="List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet")
            add_runs(p, m.group(2))
        elif m := re.match(r"^(\s*)(\d+)\.\s+(.*)", ln):
            p = doc.add_paragraph(); pf = p.paragraph_format
            pf.left_indent = Inches(0.35 + (0.3 if len(m.group(1)) >= 2 else 0)); pf.first_line_indent = Inches(-0.25)
            pf.space_after = Pt(3)
            add_runs(p, m.group(2) + ".\t" + m.group(3))
            pf.tab_stops.add_tab_stop(pf.left_indent)
        elif s.startswith(">"):
            p = doc.add_paragraph(); shade(p._p, QUOTE_FILL)
            p.paragraph_format.left_indent = Inches(0.4); add_runs(p, s.lstrip("> "))
            p.runs and setattr(p.runs[0], "italic", True)
        elif s in ("---", "***"):
            pass
        elif s:
            p = doc.add_paragraph(); add_runs(p, s)
        i += 1
    doc.save(str(dst))


def self_check():
    """python md2docx.py with no arguments: converts a sample and checks the structure."""
    import os
    import tempfile
    sample = ("# Title\n\nIntro with **bold** and `code`.\n\n## Section\n\n- one\n  - nested\n- [ ] todo\n\n"
              "1. first\n2. second\n\n| A | B |\n|---|---|\n| x | y |\n\n```\ncode line\n```\n")
    d = tempfile.mkdtemp()
    src, dst = os.path.join(d, "s.md"), os.path.join(d, "s.docx")
    with open(src, "w", encoding="utf-8") as f:
        f.write(sample)
    convert(src, dst)
    doc = Document(dst)
    styles = [p.style.name for p in doc.paragraphs]
    texts = [p.text for p in doc.paragraphs]
    assert styles[0] == "Title" and "Heading 2" in styles, styles
    assert styles.count("List Bullet") == 2 and "List Bullet 2" in styles, styles
    assert "1.\tfirst" in texts and "2.\tsecond" in texts, texts
    assert "☐ todo" in texts and "code line" in texts, texts
    t = doc.tables[0]
    assert t.cell(0, 0).text == "A" and t.cell(1, 1).text == "y" and len(t.rows) == 2
    print("md2docx self-check ok")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        self_check()
    else:
        convert(sys.argv[1], sys.argv[2])
