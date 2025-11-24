import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import geopandas as gpd
    import marimo as mo
    import polars as pl
    import rioxarray
    import xarray as xr
    import numpy as np
    import fiona
    from geocube.api.core import make_geocube
    import rasterio.features
    return gpd, mo, np, pl, rasterio, rioxarray, xr


@app.cell
def _(rioxarray):
    naturskog = rioxarray.open_rasterio(
        "C:/Users/havh/Downloads/naturskog_v1_naturskognaerhet.tif"
    )
    naturskog
    return (naturskog,)


@app.cell
def _(gpd):
    # henter kalkkartet
    kalk_data = gpd.read_file(
        "C:/Users/havh/Downloads/KalkinnholdBerggrunn.gdb",
        layer="Kalkinnhold_berggrunnN250",
    )

    #rydder data
    kalk_data_clean = kalk_data[
        (~kalk_data.geometry.is_empty) & 
        (kalk_data.geometry.notna()) &
        (kalk_data["kalkinnhold_hovedbergart"].notna())
    ]
    return (kalk_data_clean,)


@app.cell
def _(kalk_data_clean, naturskog):
    # matcher alle CRS

    kalk_data_reprojected = kalk_data_clean.to_crs(naturskog.rio.crs)
    return (kalk_data_reprojected,)


@app.cell
def _(kalk_data_reprojected, naturskog):
    # Check if CRS match
    print(f"Raster CRS: {naturskog.rio.crs}")
    print(f"Kalk_kart: {kalk_data_reprojected.crs}")
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _(mo):
    mo.md(r"""
    Legger alle verdiene til et felles raster
    """)
    return


@app.cell
def _(kalk_data_reprojected, naturskog, np, rasterio, xr):
    # Define the column name in your GDB that contains the classes (1-5)
    kalk_col_name = "kalkinnhold_hovedbergart"

    # Filter out rows with NaN values in the kalk column
    kalk_data_valid = kalk_data_reprojected[
        kalk_data_reprojected[kalk_col_name].notna()
    ]

    shapes = (
        (geom, int(value))
        for geom, value in zip(
            kalk_data_valid.geometry, kalk_data_valid[kalk_col_name]
        )
    )

    # Rasterize: Turn the vector data into a grid matching the naturskog raster
    kalk_array = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=naturskog.shape[-2:],  # (Height, Width)
        transform=naturskog.rio.transform(),
        fill=0,  # Background value
        dtype="uint8",
    )

    # Wrap in xarray with proper spatial metadata
    kalk_raster = xr.DataArray(
        kalk_array[np.newaxis, :, :],  # Add band dimension to match naturskog
        dims=["band", "y", "x"],
        coords={
            "band": [1],
            "y": naturskog.y,
            "x": naturskog.x,
        },
    )

    # Attach CRS information
    kalk_raster.rio.write_crs(naturskog.rio.crs, inplace=True)
    kalk_raster.rio.write_transform(naturskog.rio.transform(), inplace=True)

    kalk_raster
    return kalk_col_name, kalk_raster


@app.cell
def _(mo):
    mo.md(r"""
    #send mail it tromsø
    """)
    return


@app.cell
def _(kalk_col_name, kalk_raster, naturskog, pl):
    # 1. Flatten the grids into 1D arrays
    # naturskog is xarray, .data gives values. [0] selects the first band.
    naturskog_flat = naturskog.data[0].flatten()
    kalk_flat = kalk_raster[kalk_col_name].data[0].flatten()

    # 2. Combine into a Polars DataFrame
    df_combined = pl.DataFrame({"naturskog": naturskog_flat, "kalk": kalk_flat})
    return (df_combined,)


@app.cell
def _(df_combined, pl):
    # 3. Filter to keep only relevant data (remove background/0 values)
    df_clean = df_combined.filter((pl.col("naturskog") > 0) & (pl.col("kalk") > 0))
    return


if __name__ == "__main__":
    app.run()
