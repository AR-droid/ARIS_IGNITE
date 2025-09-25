# Research Notebook Feature

A smart research notebook integrated with the ARIS IGNITE platform that allows researchers to take notes, manage citations, and organize their research workflow.

## Features

- **Rich Text Editor**: Write and format research notes with a powerful WYSIWYG editor
- **Citation Management**: Search and insert citations from academic databases
- **Note Organization**: Create, update, and organize research notes
- **Export Options**: Export notes to various formats (PDF, Markdown, etc.)
- **Integration**: Seamlessly integrates with the existing ARIS IGNITE knowledge graph

## Setup

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables in a `.env` file:
   ```
   CORE_API_KEY=your_core_api_key_here
   FLASK_APP=app.py
   FLASK_ENV=development
   ```

## Usage

1. Start the development server:
   ```bash
   flask run
   ```

2. Open your browser and navigate to `http://localhost:5000/notebook`

3. Start creating and managing your research notes!

## API Endpoints

### Notes

- `GET /api/notes` - List all notes
- `POST /api/notes` - Create a new note
- `GET /api/notes/<note_id>` - Get a specific note
- `PUT /api/notes/<note_id>` - Update a note
- `DELETE /api/notes/<note_id>` - Delete a note
- `GET /api/notes/recent` - Get recently updated notes

### Citations

- `GET /api/citations` - List all saved citations
- `POST /api/citations` - Create a new citation
- `GET /api/citations/search` - Search for citations in external databases

## Integration with Knowledge Graph

The research notebook is integrated with the ARIS IGNITE knowledge graph, allowing you to:

- Link notes to concepts in the knowledge graph
- Automatically extract and suggest relevant concepts from your notes
- Visualize connections between your notes and research papers

## Development

To contribute to the development of the research notebook feature:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes and commit them
4. Push to your fork and submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
