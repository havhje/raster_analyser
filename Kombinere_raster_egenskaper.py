import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _(combined_mask, kalk_flat, mo, naturskog_flat, pl):
    # Add this diagnostic cell
    mo.md("## Coverage Analysis - CRITICAL")

    total_pixels = len(naturskog_flat)
    naturskog_valid_pixels = (naturskog_flat != 255).sum()
    kalk_valid_pixels = (kalk_flat != 0).sum()
    overlap_pixels = combined_mask.sum()

    coverage_analysis = pl.DataFrame({
        "Category": [
            "Total raster pixels",
            "Naturskog valid (classes 1-7)",
            "Kalk valid (classes 1-5)", 
            "Overlap (both valid)",
            "Naturskog WITH kalk data (%)",
            "Naturskog WITHOUT kalk data",
            "Data loss from no kalk coverage"
        ],
        "Count": [
            f"{total_pixels:,}",
            f"{naturskog_valid_pixels:,}",
            f"{kalk_valid_pixels:,}",
            f"{overlap_pixels:,}",
            f"{(overlap_pixels / naturskog_valid_pixels * 100):.2f}%",
            f"{(naturskog_valid_pixels - overlap_pixels):,}",
            f"{((naturskog_valid_pixels - overlap_pixels) / naturskog_valid_pixels * 100):.2f}%"
        ]
    })

    coverage_analysis
    return


@app.cell(column=1)
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

    # rydder data
    kalk_data_clean = kalk_data[
        (~kalk_data.geometry.is_empty)
        & (kalk_data.geometry.notna())
        & (kalk_data["kalkinnhold_hovedbergart"].notna())
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


@app.cell(column=2)
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
    return (kalk_raster,)


@app.cell
def _(kalk_raster, naturskog):
    # 1. Flatten the grids into 1D arrays
    naturskog_flat = naturskog.data[0].flatten()
    kalk_flat = kalk_raster.data[0].flatten()  # Remove [kalk_col_name] indexing
    return kalk_flat, naturskog_flat


@app.cell
def _(kalk_flat, naturskog_flat, pl):
    naturskog_mask = (naturskog_flat > 0) & (naturskog_flat != 255)
    kalk_mask = kalk_flat > 0
    combined_mask = naturskog_mask & kalk_mask

    # Apply filter to both arrays
    naturskog_filtered = naturskog_flat[combined_mask]
    kalk_filtered = kalk_flat[combined_mask]

    # Now create DataFrame with filtered data
    kalk_naturskog_clean = pl.DataFrame(
        {"naturskog": naturskog_filtered, "kalk": kalk_filtered}
    )

    kalk_naturskog_clean
    return combined_mask, kalk_naturskog_clean


@app.cell
def _(kalk_naturskog_clean, naturskog, pl):
    # Extract pixel dimensions
    pixel_width = abs(naturskog.rio.resolution()[0])
    pixel_height = abs(naturskog.rio.resolution()[1])
    pixel_area_m2 = pixel_width * pixel_height

    # Calculate areas for each naturskog-kalk combination
    area_summary = (
        kalk_naturskog_clean.group_by(["naturskog", "kalk"])
        .agg(pl.len().alias("pixel_count"))
        .with_columns(
            [
                (pl.col("pixel_count") * pixel_area_m2).alias("area_m2"),
                (pl.col("pixel_count") * pixel_area_m2 / 10_000).alias("area_ha"),
                (pl.col("pixel_count") * pixel_area_m2 / 1_000_000).alias(
                    "area_km2"
                ),
            ]
        )
        .sort(["kalk", "naturskog"])
    )

    area_summary
    return (area_summary,)


@app.cell(column=3)
def _(mo):
    mo.md(r"""
    ## Visualiserer
    """)
    return


@app.cell
def _(area_summary, pl):
    # Add percentage column to area_summary
    area_summary_pct = area_summary.with_columns(
        (pl.col("area_km2") / pl.col("area_km2").sum() * 100).alias("percentage")
    )

    area_summary_pct
    return (area_summary_pct,)


@app.cell
def _(area_summary, area_summary_pct):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create pivot tables
    area_km2_matrix = (
        area_summary.to_pandas()
        .pivot(index="kalk", columns="naturskog", values="area_km2")
        .fillna(0)
    )

    percentage_matrix = (
        area_summary_pct.to_pandas()
        .pivot(index="kalk", columns="naturskog", values="percentage")
        .fillna(0)
    )

    # Create figure with two subplots
    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Left plot: Absolute area
    sns.heatmap(
        area_km2_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Areal (km²)"},
        ax=_ax1,
    )
    _ax1.invert_yaxis()
    _ax1.set_title("Areal i km²", fontsize=13, pad=15)
    _ax1.set_xlabel("Naturskogsnærhet klasse", fontsize=11)
    _ax1.set_ylabel("Kalkinnhold klasse", fontsize=11)

    # Right plot: Percentage
    sns.heatmap(
        percentage_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Prosentandel (%)"},
        ax=_ax2,
    )
    _ax2.invert_yaxis()
    _ax2.set_title("Prosentandel", fontsize=13, pad=15)
    _ax2.set_xlabel("Naturskogsnærhet klasse", fontsize=11)
    _ax2.set_ylabel("Kalkinnhold klasse", fontsize=11)

    plt.suptitle(
        "Naturskog fordelt på kalkinnhold i berggrunn",
        fontsize=14,
        y=1.02
    )

    plt.tight_layout()

    plt.gca()
    return


if __name__ == "__main__":
    app.run()
