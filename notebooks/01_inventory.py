# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "pillow"]
# ///
"""Step 1: inventory every extracted asset and decide what to work on first."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from collections import Counter

    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir().parent))
    from ca_pipeline import manifest
    return Counter, manifest, mo


@app.cell
def _(mo):
    root = mo.ui.text(value=str(mo.notebook_dir().parent / "assets" / "converted"),
                      label="Converted assets folder", full_width=True)
    root
    return (root,)


@app.cell
def _(manifest, mo, root):
    records = manifest.scan(root.value)
    out_csv = mo.notebook_dir().parent / "assets" / "manifest.csv"
    manifest.write_csv(records, out_csv)
    mo.md(f"**{len(records)} files** scanned, manifest written to `{out_csv}`")
    return (records,)


@app.cell
def _(Counter, mo, records):
    by_cat = Counter((r.priority, r.category, r.kind) for r in records)
    summary = [{"priority": p, "category": c, "kind": k, "files": n}
               for (p, c, k), n in sorted(by_cat.items())]
    needs_convert = sum(1 for r in records if r.note)
    mo.vstack([
        mo.md("### By category (work top to bottom)"),
        mo.ui.table(summary, selection=None),
        mo.callout(mo.md(f"{needs_convert} files still need converting or are unreadable "
                         "(see `note` column)."), kind="warn") if needs_convert else mo.md(""),
    ])
    return


@app.cell
def _(mo, records):
    mo.ui.table([r.__dict__ for r in records], page_size=25)
    return


if __name__ == "__main__":
    app.run()
