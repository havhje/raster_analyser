import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import rioxarray
    import polars as pl
    import numpy as np
    import altair as alt
    from rasterio.enums import Resampling
    return Resampling, alt, mo, np, pl, rioxarray


@app.cell
def _(mo):
    mo.md(r"""
    ### Naturskogsnærhet
    """)
    return


@app.cell
def _(rioxarray):
    naturskog = rioxarray.open_rasterio(
        "/Users/havardhjermstad-sollerud/Downloads/naturskog_v1_naturskognaerhet.tif"
    )

    naturskog
    return (naturskog,)


@app.cell
def _(naturskog):
    # Mask NoData values before plotting
    naturskog_masked = naturskog.where(naturskog != 255)

    naturskog_masked.hvplot.image(
        x="x",
        y="y",
        rasterize=True,
        cmap="tab10",
        clim=(1, 7),
        aspect="equal",
        frame_height=800,
        title="Naturskog",
    )
    return


@app.cell
def _():
    from localtileserver import TileClient, get_folium_tile_layer
    import folium

    # Create tile client for naturskog raster
    naturskog_client = TileClient(
        "/Users/havardhjermstad-sollerud/Downloads/naturskog_v1_naturskognaerhet.tif"
    )

    # Create folium map centered on the data
    naturskog_folium_map = folium.Map(
        location=naturskog_client.center(),
        zoom_start=naturskog_client.default_zoom,
        tiles="OpenStreetMap",
    )

    # Add tile layer with categorical colormap
    naturskog_folium_layer = get_folium_tile_layer(
        naturskog_client,
        colormap="tab10",  # Categorical colormap with distinct colors
        vmin=1,
        vmax=7,  # Only show classes 0-7, exclude NoData (255)
        opacity=0.8,
        attr="Naturskog",
    )

    naturskog_folium_map.add_child(naturskog_folium_layer)

    naturskog_folium_map
    return


@app.cell
def _(naturskog):
    # Check if NoData is set
    naturskog.rio.nodata
    return


@app.cell
def _(naturskog):
    # Get pixel resolution (width and height)
    naturskog.rio.resolution()
    return


@app.cell
def _(naturskog, np, pl):
    naturskog_verdi = naturskog.values.flatten()

    naturskog_verdi_clean = naturskog_verdi[naturskog_verdi != 255]

    unike_klasser, counts = np.unique(naturskog_verdi_clean, return_counts=True)

    klasse_distribusjon = pl.DataFrame(
        {"Klasse": unike_klasser, "Pixel_Count": counts}
    )
    return klasse_distribusjon, naturskog_verdi


@app.cell
def _(alt, klasse_distribusjon, mo):
    fordeling = (
        alt.Chart(klasse_distribusjon)
        .mark_bar()
        .encode(
            x=alt.X("Klasse:O"), y=alt.Y("Pixel_Count:Q", title="Antall piksler")
        )
        .properties(
            title="Fordeling av klasser i naturskogsnærhetkartet",
            width=600,
            height=400,
        )
    )

    mo.ui.altair_chart(fordeling)
    return


@app.cell(column=1)
def _(mo):
    mo.md(r"""
    ### Valgt kartlag
    """)
    return


@app.cell
def _(rioxarray):
    hakkespett = rioxarray.open_rasterio(
        "/Users/havardhjermstad-sollerud/Downloads/trua-lav-hotspots_norge_2025_32633.tiff"
    )
    hakkespett
    return (hakkespett,)


@app.cell
def _(hakkespett):
    hakkespett.rio.resolution()
    return


@app.cell
def _(hakkespett, np, pl):
    hakkespett_verdi = hakkespett.values.flatten()

    hakkespett_verdi_clean = hakkespett_verdi[~np.isnan(hakkespett_verdi)]

    hakkespett_distribusjon = pl.DataFrame({"Verdi": hakkespett_verdi_clean})
    return (hakkespett_distribusjon,)


@app.cell
def _(alt, hakkespett_distribusjon, mo):
    # Create histogram
    hakkespett_fordeling = (
        alt.Chart(hakkespett_distribusjon)
        .mark_bar()
        .encode(
            x=alt.X(
                "Verdi:Q", bin=alt.Bin(maxbins=50), title="Hakkespett-rikhet (0-1)"
            ),
            y=alt.Y("count()", title="Antall piksler"),
            tooltip=[
                alt.Tooltip("Verdi:Q", bin=alt.Bin(maxbins=50), title="Verdi"),
                alt.Tooltip("count()", title="Antall"),
            ],
        )
        .properties(
            title="Fordeling av hakkespett-rikhet",
            width=600,
            height=400,
        )
    )

    mo.ui.altair_chart(hakkespett_fordeling)
    return


@app.cell(column=2)
def _(mo):
    mo.md(r"""
    ### Resampling
    """)
    return


@app.cell
def _(Resampling, naturskog):
    naturskog_500m = naturskog.rio.reproject(
        naturskog.rio.crs, resolution=(500, 500), resampling=Resampling.mode
    )
    return (naturskog_500m,)


@app.cell
def _(naturskog_500m, naturskog_verdi):
    naturskog_500m_verdi = naturskog_500m.values.flatten()

    naturskog_500m_verdi_clean = naturskog_verdi[naturskog_verdi != 255]
    return


@app.cell
def _(Resampling, hakkespett, naturskog_500m):
    hakkespett_aligned = hakkespett.rio.reproject_match(
        naturskog_500m,
        resampling=Resampling.bilinear,  # Good for continuous data
    )
    return (hakkespett_aligned,)


@app.cell
def _(hakkespett_aligned, naturskog_500m, pl):
    # Compare spatial properties
    comparison = pl.DataFrame(
        {
            "Property": ["CRS", "X Resolution", "Y Resolution", "Width", "Height"],
            "Naturskog": [
                str(naturskog_500m.rio.crs),
                str(naturskog_500m.rio.resolution()[0]),
                str(naturskog_500m.rio.resolution()[1]),
                str(naturskog_500m.rio.width),
                str(naturskog_500m.rio.height),
            ],
            "Hakkespett": [
                str(hakkespett_aligned.rio.crs),
                str(hakkespett_aligned.rio.resolution()[0]),
                str(hakkespett_aligned.rio.resolution()[1]),
                str(hakkespett_aligned.rio.width),
                str(hakkespett_aligned.rio.height),
            ],
        }
    )

    comparison
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Korrelasjon
    """)
    return


@app.cell
def _(hakkespett_aligned, naturskog_500m, np, pl):
    # 1. Select first band and flatten
    naturskog_flat = naturskog_500m[0].values.flatten()
    hakkespett_flat = hakkespett_aligned[0].values.flatten()

    # 2. Remove NaN and NoData (255)
    valid_mask = (~np.isnan(hakkespett_flat)) & (naturskog_flat != 255)
    naturskog_valid = naturskog_flat[valid_mask]
    hakkespett_valid = hakkespett_flat[valid_mask]

    # 3. Create dataframe (public variable)
    korrelasjon_data = pl.DataFrame(
        {
            "Naturskog_klasse": naturskog_valid.astype(int),
            "Hakkespett_rikhet": hakkespett_valid,
        }
    )

    # 4. Calculate summary statistics per class
    statistikk_per_klasse = (
        korrelasjon_data.group_by("Naturskog_klasse")
        .agg(
            [
                pl.col("Hakkespett_rikhet").mean().alias("Gjennomsnitt"),
                pl.col("Hakkespett_rikhet").std().alias("Std_avvik"),
                pl.col("Hakkespett_rikhet").count().alias("Antall_piksler"),
            ]
        )
        .sort("Naturskog_klasse")
    )

    statistikk_per_klasse
    return (korrelasjon_data,)


@app.cell
def _(korrelasjon_data, pl):
    statistikk_to_delt = (
        korrelasjon_data.with_columns(
            pl.when(pl.col("Naturskog_klasse") == 1)
            .then(pl.lit("Klasse 1"))
            .otherwise(pl.lit("Klasse 2-7"))
            .alias("Klasse_gruppe")
        )
        .group_by("Klasse_gruppe")
        .agg(
            [
                pl.col("Hakkespett_rikhet").mean().alias("Gjennomsnitt"),
                pl.col("Hakkespett_rikhet").std().alias("Std_avvik"),
                pl.col("Hakkespett_rikhet").count().alias("Antall_piksler"),
            ]
        )
        .sort("Klasse_gruppe")
    )

    statistikk_to_delt
    return


if __name__ == "__main__":
    app.run()
