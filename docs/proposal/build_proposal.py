"""Build and verify the two-page US Letter proposal using installed LaTeX."""
from pathlib import Path
import shutil
import subprocess
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'docs/proposal/latex/main.tex'
OUTPUT = ROOT / 'output/pdf/shopping-budget-advisor-proposal.pdf'
BUILD = ROOT / 'tmp/pdfs/latex-build'


def build():
    compiler = shutil.which('pdflatex')
    if not compiler:
        raise SystemExit('pdflatex with newtxtext and newtxmath is required')
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, '-no-shell-escape', '-interaction=nonstopmode',
               '-halt-on-error', '-file-line-error', f'-output-directory={BUILD}', SOURCE.name]
    for _ in range(2):
        result = subprocess.run(command, cwd=SOURCE.parent, capture_output=True, text=True,
                                errors='replace')
        if result.returncode:
            raise SystemExit(result.stdout[-5000:] + result.stderr[-1000:])
    draft = BUILD / 'main.pdf'
    reader = PdfReader(draft)
    assert len(reader.pages) == 2, f'Expected two pages, got {len(reader.pages)}'
    for page in reader.pages:
        assert abs(float(page.mediabox.width) - 612) < 0.1
        assert abs(float(page.mediabox.height) - 792) < 0.1
    page_text = [p.extract_text() for p in reader.pages]
    assert all(term in '\n'.join(page_text) for term in
               ['Motivation and task', 'Proposed system', 'Evaluation plan', 'Bolin CHEN'])
    assert 'Figure1' in ''.join(page_text[0].split())
    assert 'Table1' in ''.join(page_text[1].split())
    source = SOURCE.read_text(encoding='utf-8')
    assert r'\usepackage{newtxtext}' in source and r'\usepackage{newtxmath}' in source
    assert r'\documentclass[11pt,letterpaper]' in source
    log = (BUILD / 'main.log').read_text(encoding='utf-8', errors='replace')
    assert 'Overfull' not in log, 'LaTeX reports overfull content; inspect main.log'
    shutil.copy2(draft, OUTPUT)
    print(f'Created {OUTPUT}: two US Letter pages; newtx fonts; 11 pt body; layout checks passed.')


if __name__ == '__main__':
    build()
