Here’s a clean, **practical README section** you can drop into your repo (e.g. `docs/README.md` or main `README.md`).

---

# 📚 Documentation Guide

This project uses **Sphinx + MyST (Markdown)** to build documentation, and supports both:

* human-written guides (`.md`)
* auto-generated API reference (`.rst`)

---

# 🧱 Documentation structure

```text
docs/
  source/                 ← ALL documentation source files
    index.rst             ← root entrypoint
    profiling_foundation.md
    search_iteration_architecture.md
    debug_live_session.md
    api/                  ← auto-generated API docs
  build/                  ← generated HTML (do not edit)
  conf.py
  Makefile
```

👉 **Important rule:**
All documentation files must live under:

```text
docs/source/
```

If a file is outside this folder, Sphinx will not see it.

---

# ⚙️ One-time setup

Install documentation dependencies:

```bash
pip install -e .[docs]
```

or manually:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

---

# 🧪 Build the documentation locally

From the `docs/` directory:

```bash
make clean
make html
xdg-open build/html/index.html
```

👉 Output is generated in:

```text
docs/build/html/
```

---

# ✍️ Writing and updating docs

## 1. Create or edit a page

Create a new Markdown file in:

```text
docs/source/
```

Example:

```bash
vim docs/source/my_new_page.md
```

Start with a top-level heading:

```md
# My New Page
```

---

## 2. Add it to the documentation tree

Edit:

```text
docs/source/index.rst
```

Add your page to a `toctree`:

```rst
.. toctree::
   :maxdepth: 2

   my_new_page
```

⚠️ Do NOT include `.md` in the name.

---

## 3. Rebuild docs

```bash
make html
```

---

# 🤖 Updating API documentation

When Python APIs change, regenerate API docs:

From repo root:

```bash
rm -rf docs/source/api
mkdir -p docs/source/api
sphinx-apidoc -f -o docs/source/api src/anemone
```

Then rebuild:

```bash
cd docs
make html
```

---

# 🧠 Key concepts

### Sphinx only shows what is in the toctree

If a page is not listed in a `toctree`, it will not appear in the docs.

---

### Markdown support comes from MyST

We use MyST to allow `.md` files inside Sphinx.

---

### Separate concerns

* `.md` → guides, tutorials, architecture
* `api/*.rst` → generated API reference

Do not mix them.

---

# 🚀 Recommended workflow

### When writing features (e.g. profiling)

1. update or create:

   * `docs/source/profiling_foundation.md`
2. ensure it is in `index.rst`
3. rebuild docs

---

### When modifying APIs

1. regenerate API docs
2. rebuild docs

---

# ⚠️ Common issues

### ❌ “document not found”

* file is not in `docs/source/`
* or not listed in `toctree`

---

### ❌ Markdown not rendering

* `myst-parser` not installed

---

### ❌ Theme error

* install:

  ```bash
  pip install sphinx-rtd-theme
  ```

---

# 🧹 Optional cleanup

To remove the warning:

```text
html_static_path entry '_static' does not exist
```

Run:

```bash
mkdir -p docs/source/_static
```

---

# ✅ Summary

The workflow is:

```bash
# edit docs
vim docs/source/*.md

# rebuild
cd docs
make html
xdg-open build/html/index.html
```

and occasionally:

```bash
# regenerate API docs
sphinx-apidoc -f -o docs/source/api src/anemone
```

---

If you want, I can next:

* add a polished `index.rst` with sections (Guides / Profiling / API)
* or turn your profiling doc into a proper “Getting Started” page for users
