# Rastr

This is the Rasterizing filter for [nodesignfoundry.com](https://nodesignfoundry.com)

It gets hinted outlines from a hinted file and turns it into a font.


# CLI
- Ensure you have Python 3 installed by running `python -v` or `python3 -v` in your terminal.
- Verify that `git` is installed by executing `git -v` in your terminal. If the git version is displayed, you're all set. Otherwise, proceed with installing git.
- Install the plugin for Python using the following command: `python -m pip install git+https://github.com/no-design-foundry/filters-rasterizer.git`
- You run the tool in terminal via `rasterizer <hinted font> <font size>`. This will rasterize the font and output `.ufo` file next to your input file.
- There is additional argument `--output-dir` that will save the `.ufo` file to given folder

## TODO

- [ ] Find a way how to remove `scipy` dependency (numpy version was too slow)
- [x] Turn this into a CLI 

