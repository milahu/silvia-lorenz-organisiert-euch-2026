#!/bin/sh

# TODO set config values
cover_src=060-rotate-crop/133.tiff

magick "$cover_src" -scale 50% -quality 50% cover.avif
