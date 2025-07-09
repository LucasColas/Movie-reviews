import nbformat
from pathlib import Path

nb_path = Path("IMDB_movie_reviews.ipynb")
nb = nbformat.read(nb_path, as_version=4)

# Remove all cell.metadata.widgets entries
for cell in nb.cells:
    if "widgets" in cell.metadata:
        del cell.metadata["widgets"]

nbformat.write(nb, nb_path)
print("Cleaned:", nb_path)
