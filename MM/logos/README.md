# Poster logo assets

All four university marks used by `poster.tex` are vector PDFs. The PDFs preserve
the paths and text from their EPS/PDF sources; they are not bitmap images wrapped
in PDF containers.

| File used in poster | Source | Original format |
| --- | --- | --- |
| `fudan-emblem.pdf` | Fudan University official identity download | EPS |
| `pku-emblem.pdf` | Peking University Visual Identity Office download | EPS |
| `sjtu-emblem.pdf` | `sjtutex` visual-identity assets | PDF |
| `tongji-logo.pdf` | Tongji University official image portal download | PDF |

Source archives are retained in `source_archives/`, and the extracted originals
are retained in `extracted/` for traceability.

`poster.tex` clips the official source artboards to display only each circular emblem.
Clipping does not rasterize the artwork. The horizontal lockups remain in this folder
as optional assets but are no longer used in the poster header.

The ACM Multimedia conference mark supplied with the poster package is a PNG, so
it remains raster. Moving or converting it to a PDF container would not make it
vector.
