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


@app.cell
def _(mo):
    mo.md(r"""
    ### Hakkespetter
    """)
    return


@app.cell
def _(rioxarray):
    hakkespett = rioxarray.open_rasterio(
        "/Users/havardhjermstad-sollerud/Downloads/alle-hakkespett-rikhet_norge_2025_32633.tiff"
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


@app.cell(column=1)
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
def _(naturskog_500m):
    naturskog_500m.rio.resolution()
    return


@app.cell
def _(naturskog_500m, naturskog_verdi):
    naturskog_500m_verdi = naturskog_500m.values.flatten()

    naturskog_500m_verdi_clean = naturskog_verdi[naturskog_verdi != 255]
    return


@app.cell
def _(hakkespett, naturskog_500m, pl):
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
                str(hakkespett.rio.crs),
                str(hakkespett.rio.resolution()[0]),
                str(hakkespett.rio.resolution()[1]),
                str(hakkespett.rio.width),
                str(hakkespett.rio.height),
            ],
        }
    )

    comparison
    return


if __name__ == "__main__":
    app.run()
