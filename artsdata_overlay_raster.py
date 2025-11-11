import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import rioxarray
    import numpy as np
    import geopandas as gpd
    return gpd, mo, rioxarray


@app.cell
def _(rioxarray):
    naturskog = rioxarray.open_rasterio("C:/Users/havh/Downloads/naturskog_v1_naturskognaerhet.tif")
    naturskog
    return (naturskog,)


@app.cell
def _(naturskog):
    print(f"CRS: {naturskog.rio.crs}")
    print(f"EPSG Code: {naturskog.rio.crs.to_epsg()}")
    print(f"WKT: {naturskog.rio.crs.to_wkt()}")
    return


@app.cell
def _(naturskog):
    import hvplot.xarray

    naturskog.hvplot.image(
        x='x', 
        y='y',
        rasterize=True,
        cmap='Category10',  # or 'Set2', 'Pastel1', 'Dark2'
        aspect='equal',
        frame_width=600,
        clabel='Category',
        colorbar=True
    )
    return


@app.cell
def _():
    import os
    # Set environment variable to allow unlimited GeoJSON size
    os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'
    return


@app.cell
def _(gpd):
    # Read the first layer (polygons/areas)
    naturtype_omr = gpd.read_file(
        "C:/Users/havh/Downloads/Naturtyper_nin_0000_norge_4326_GEOJSON.json",
        layer='Naturtype_nin_omr'
    )
    naturtype_omr
    return (naturtype_omr,)


@app.cell
def _(gpd):
    # Read the second layer (coverage)
    naturtype_dekning = gpd.read_file(
        "C:/Users/havh/Downloads/Naturtyper_nin_0000_norge_4326_GEOJSON.json",
        layer='Naturtyper_nin_dekning'
    )
    naturtype_dekning
    return (naturtype_dekning,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _(mo):
    mo.md(r"""
    ### Klipper
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(naturskog, naturtype_dekning, naturtype_omr):
    # Check if CRS match
    print(f"Raster CRS: {naturskog.rio.crs}")
    print(f"GeoJSON 1 CRS: {naturtype_omr.crs}")
    print(f"GeoJSON 2 CRS: {naturtype_dekning.crs}")
    return


@app.cell
def _(naturskog, naturtype_dekning, naturtype_omr):
    # Reproject GeoDataFrames to match raster CRS
    naturtype_omr_projected = naturtype_omr.to_crs(naturskog.rio.crs)
    naturtype_dekning_projected = naturtype_dekning.to_crs(naturskog.rio.crs)
    return naturtype_dekning_projected, naturtype_omr_projected


@app.cell
def _(naturskog, naturtype_dekning_projected):
    # Get bounding box of naturtype_dekning
    bounds = naturtype_dekning_projected.total_bounds  # (minx, miny, maxx, maxy)

    # Crop to bounding box first (much faster and less memory)
    naturskog_cropped = naturskog.rio.clip_box(*bounds)
    naturskog_cropped
    return (naturskog_cropped,)


@app.cell
def _(naturskog_cropped, naturtype_dekning_projected):
    # Now clip the smaller raster to preserve only naturtype_dekning areas
    naturskog_clipped = naturskog_cropped.rio.clip(
        naturtype_dekning_projected.geometry,
        all_touched=False,
        drop=True
    )
    naturskog_clipped
    return (naturskog_clipped,)


@app.cell
def _(naturskog_clipped, naturtype_omr_projected):
    from rasterio.features import geometry_mask

    # Create mask where True = areas to KEEP
    mask = geometry_mask(
        naturtype_omr_projected.geometry,
        out_shape=(naturskog_clipped.sizes['y'], naturskog_clipped.sizes['x']),
        transform=naturskog_clipped.rio.transform(),
        invert=True
    )

    # Apply mask
    naturskog_final = naturskog_clipped.where(mask)
    naturskog_final
    return


if __name__ == "__main__":
    app.run()
