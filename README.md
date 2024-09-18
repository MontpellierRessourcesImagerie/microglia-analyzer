# Microglia Analyzer

![GitHub License](https://img.shields.io/github/license/MontpellierRessourcesImagerie/microglia-analyzer)
![Python Version](https://img.shields.io/badge/Python-3.9-blue?logo=python)

A Napari plugin allowing to detect and segment microglia on fluorescent images.

It consists in:
- Detecting the microglia with a YOLOv5 model.
- Segmenting them with a UNet model.
- Using some morphology to extract metrics such as:
    - The total length
    - The length of the longest path
    - The number of leaves
    - The number of vertices
    - Area of the convex hull
    - Solidity/extent

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `microglia-analyzer` via [pip]:

    pip install microglia-analyzer



To install latest development version :

    pip install git+https://github.com/MontpellierRessourcesImagerie/microglia-analyzer.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"microglia-analyzer" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/MontpellierRessourcesImagerie/microglia-analyzer/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
