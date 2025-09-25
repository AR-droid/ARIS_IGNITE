# LaTeX Integration for ARIS IGNITE

This document outlines the LaTeX integration features added to the ARIS IGNITE notebook system.

## Features

1. **LaTeX Export**
   - Export notes to LaTeX format
   - Generate PDFs from LaTeX source
   - Preview LaTeX output before downloading

2. **Equation Editor**
   - Insert mathematical equations using natural language
   - Automatic conversion of simple expressions to LaTeX
   - Support for complex LaTeX equations

3. **Table and Chart Generation**
   - Convert data to LaTeX tables
   - Generate charts and include them as LaTeX figures

4. **Live Preview**
   - Preview LaTeX output in real-time
   - View rendered equations and formatting

## Installation

Make sure you have the following dependencies installed:

```bash
# Required system dependencies
sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra

# Python dependencies (automatically installed via requirements.txt)
pip install -r requirements.txt
```

## Usage

### Exporting to LaTeX

1. Open a note in the notebook
2. Click the "Export" dropdown in the toolbar
3. Select "Export to LaTeX" to download the LaTeX source
4. Select "Generate PDF via LaTeX" to generate a PDF directly

### Inserting Equations

1. Place your cursor where you want to insert an equation
2. Click "Insert" > "Equation" in the toolbar
3. Enter your equation in natural language (e.g., "integral of x squared from 0 to 1")
4. The equation will be converted to LaTeX and inserted into your note

### Using Tables and Charts

1. Format your data as a table in the note
2. The LaTeX exporter will automatically convert it to a LaTeX table
3. For charts, use the chart insertion tool to create visualizations

## API Endpoints

- `POST /api/notebook/export/latex/<note_id>` - Export a note to LaTeX format
- `POST /api/notebook/export/latex/pdf/<note_id>` - Generate PDF from LaTeX
- `GET /api/notebook/download/latex/<note_id>` - Download LaTeX source
- `GET /api/notebook/download/pdf/<note_id>` - Download generated PDF
- `POST /api/notebook/api/latex/convert` - Convert natural language to LaTeX

## Customization

You can customize the LaTeX output by modifying the templates in `latex_utils.py`. The following components can be customized:

- Document class and packages
- Page layout and margins
- Section formatting
- Table and figure styles

## Troubleshooting

### Common Issues

1. **LaTeX not installed**
   - Ensure you have a LaTeX distribution installed (e.g., TeX Live)
   - Verify that `pdflatex` is in your system PATH

2. **Missing fonts**
   - Install additional fonts if needed: `sudo apt-get install texlive-fonts-extra`

3. **PDF generation fails**
   - Check the server logs for LaTeX compilation errors
   - Ensure all required LaTeX packages are installed

## License

This feature is part of the ARIS IGNITE project and is licensed under the same terms.
