# Rastr

This is the Rasterizing filter for [nodesignfoundry.com](https://nodesignfoundry.com)

It gets hinted outlines from a hinted file and turns it into a font.


# CLI
- Install it with `python -m pip install ndf_rasterizer`
- You run the tool in terminal via `ndf_rasterizer <hinted font> <font size>`. This will rasterize the font and output `.ufo` file next to your input file.
- There is additional argument `--output-dir` that will save the `.ufo` file to given folder

## TODO

- [ ] Find a way how to remove `scipy` dependency (numpy version was too slow)
- [x] Turn this into a CLI 

