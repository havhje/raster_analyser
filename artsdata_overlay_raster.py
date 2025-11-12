import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import geopandas as gpd
    import marimo as mo
    import polars as pl
    import rioxarray
    import numpy as np
    return gpd, mo, np, pl, rioxarray


@app.cell
def _(mo):
    mo.md(r"""
    ## Importerer data
    """)
    return


@app.cell
def _(rioxarray):
    naturskog = rioxarray.open_rasterio(
        "C:/Users/havh/Downloads/naturskog_v1_naturskognaerhet.tif"
    )
    naturskog
    return (naturskog,)


@app.cell
def _(naturskog):
    print(f"CRS: {naturskog.rio.crs}")
    print(f"EPSG Code: {naturskog.rio.crs.to_epsg()}")
    print(f"WKT: {naturskog.rio.crs.to_wkt()}")
    return


@app.cell
def _():
    import os

    # Set environment variable to allow unlimited GeoJSON size
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
    return


@app.cell
def _(gpd):
    # Read the first layer (polygons/areas)
    naturtype_omr = gpd.read_file(
        "C:/Users/havh/Downloads/Naturtyper_nin_0000_norge_4326_GEOJSON.json",
        layer="Naturtype_nin_omr",
    )
    naturtype_omr
    return (naturtype_omr,)


@app.cell
def _(gpd):
    # Read the second layer (coverage)
    naturtype_dekning = gpd.read_file(
        "C:/Users/havh/Downloads/Naturtyper_nin_0000_norge_4326_GEOJSON.json",
        layer="Naturtyper_nin_dekning",
    )
    naturtype_dekning
    return (naturtype_dekning,)


@app.cell
def _(naturskog, naturtype_dekning, naturtype_omr):
    # Reproject GeoDataFrames to match raster CRS
    naturtype_omr_projected = naturtype_omr.to_crs(naturskog.rio.crs)
    naturtype_dekning_projected = naturtype_dekning.to_crs(naturskog.rio.crs)

    # Check if CRS match
    print(f"Raster CRS: {naturskog.rio.crs}")
    print(f"GeoJSON 1 CRS: {naturtype_omr_projected.crs}")
    print(f"GeoJSON 2 CRS: {naturtype_dekning_projected.crs}")
    return naturtype_dekning_projected, naturtype_omr_projected


@app.cell
def _():
    return


@app.cell(column=1)
def _(mo):
    mo.md(r"""
    ## Henter verdier til artsobspunktene (burde bruke geowombat, men må installere via conda-forge, har ikke tid nå)
    """)
    return


@app.cell
def _(pl):
    arts_obs = pl.read_parquet(
        "C:/Users/havh/OneDrive - Multiconsult/Dokumenter/Oppdrag/FoUI SVV/sopp_lav_plantea_ink_moser.parquet"
    )
    return (arts_obs,)


@app.cell
def _(arts_obs, pl):
    import rasterio
    from pyproj import Transformer

    # Open raster with rasterio
    with rasterio.open(
        "C:/Users/havh/Downloads/naturskog_v1_naturskognaerhet.tif"
    ) as src:
        # Create transformer
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        # Transform coordinates
        x_coords, y_coords = transformer.transform(
            arts_obs["decimallongitude"].to_numpy(),
            arts_obs["decimallatitude"].to_numpy(),
        )

        # Sample raster at points - returns generator
        coords = list(zip(x_coords, y_coords))
        values = [val[0] for val in src.sample(coords)]

    # Add to dataframe
    arts_obs_with_values = arts_obs.with_columns(
        pl.Series("naturskog_value", values)
    )

    arts_obs_with_values
    return (arts_obs_with_values,)


@app.cell
def _(arts_obs_with_values, gpd):
    # Convert arts_obs to GeoDataFrame and reproject to EPSG:25833 (er den alle bruker nå)
    arts_obs_gdf = gpd.GeoDataFrame(
        arts_obs_with_values.to_pandas(),
        geometry=gpd.points_from_xy(
            arts_obs_with_values["decimallongitude"],
            arts_obs_with_values["decimallatitude"],
        ),
        crs="EPSG:4326",  # Original CRS (lat/lon)
    ).to_crs("EPSG:25833")  # Reproject to match the naturtype layers
    return (arts_obs_gdf,)


@app.cell
def _(arts_obs_gdf, naturtype_dekning_projected, naturtype_omr_projected):
    # Convert geometry columns to WKT format for DuckDB
    arts_obs_for_duckdb = arts_obs_gdf.copy()
    arts_obs_for_duckdb["geometry_wkt"] = arts_obs_gdf.geometry.to_wkt()
    arts_obs_for_duckdb = arts_obs_for_duckdb.drop(columns=["geometry"])

    dekning_for_duckdb = naturtype_dekning_projected.copy()
    dekning_for_duckdb["geometry_wkt"] = (
        naturtype_dekning_projected.geometry.to_wkt()
    )
    dekning_for_duckdb = dekning_for_duckdb.drop(columns=["geometry"])

    omr_for_duckdb = naturtype_omr_projected.copy()
    omr_for_duckdb["geometry_wkt"] = naturtype_omr_projected.geometry.to_wkt()
    omr_for_duckdb = omr_for_duckdb.drop(columns=["geometry"])
    return arts_obs_for_duckdb, dekning_for_duckdb, omr_for_duckdb


@app.cell
def _(arts_obs_for_duckdb, dekning_for_duckdb, mo, omr_for_duckdb):
    arter_riktig_df = mo.sql(
        f"""
        INSTALL spatial;
        LOAD spatial;

        WITH points_in_dekning AS (
            SELECT DISTINCT a.*
            FROM arts_obs_for_duckdb a, dekning_for_duckdb d
            WHERE ST_Within(
                ST_GeomFromText(a.geometry_wkt), 
                ST_GeomFromText(d.geometry_wkt)
            )
        ),
        points_in_omr AS (
            SELECT DISTINCT a.*
            FROM arts_obs_for_duckdb a, omr_for_duckdb o
            WHERE ST_Within(
                ST_GeomFromText(a.geometry_wkt), 
                ST_GeomFromText(o.geometry_wkt)
            )
        )
        SELECT * FROM points_in_dekning
        EXCEPT
        SELECT * FROM points_in_omr;
        """
    )
    return (arter_riktig_df,)


@app.cell(column=2)
def _(mo):
    mo.md(r"""
    ### Statestikik på arter som er innenfor dekningsområdet, men utenfor MI = Øvrig natur
    """)
    return


@app.cell
def _(naturskog, np, pl):
    unike_klasser, counts = np.unique(naturskog.values.flatten(), return_counts=True)
    klasse_distribusjon = pl.DataFrame({"Klasse": unike_klasser, "Pixel_Count": counts})

    klasse_distribusjon
    return (klasse_distribusjon,)


@app.cell
def _():
    mapping = {
        255: "Ikke skog",
        1: "Skog",
        2: "2-N",
        3: "3-N",
        4: "4-N",
        5: "5-N",
        6: "6-N",
        7: "7-N",
    }
    return (mapping,)


@app.cell
def _(klasse_distribusjon, mapping, pl):
    # Rename class 255 to 0 and group to combine pixel counts
    klasse_dist_renamed = (
        klasse_distribusjon.with_columns(
            pl.col("Klasse").replace_strict(mapping, default=pl.col("Klasse"))
        )
        .group_by("Klasse")
        .agg(pl.col("Pixel_Count").sum())
    )

    klasse_dist_renamed
    return (klasse_dist_renamed,)


@app.cell
def _(arter_riktig_df, mapping, pl):
    # Group by naturskog_value and Kategori 2021, count observations
    grouped_data_med_ikke_skog = (
        arter_riktig_df.filter(~pl.col("Kategori 2021").is_in(["DD", "NT°"]))
        .group_by(["naturskog_value", "Kategori 2021"])
        .agg(pl.len().alias("count"))
        .sort(["naturskog_value", "Kategori 2021"])
        .with_columns(
            pl.col("naturskog_value").replace_strict(
                mapping, default=pl.col("naturskog_value")
            )
        )
    )
    grouped_data_med_ikke_skog
    return (grouped_data_med_ikke_skog,)


@app.cell
def _(grouped_data_med_ikke_skog, pl):
    ## Tar bort ikke skog
    grouped_data = grouped_data_med_ikke_skog.filter(
        pl.col("naturskog_value") != "Ikke skog"
    )
    return (grouped_data,)


@app.cell
def _(grouped_data, klasse_dist_renamed, pl):
    # Join with pixel counts and calculate density (observations per km²)
    # 1 pixel = 16m x 16m = 256 m²
    # 1 km² = 1,000,000 m² = 3906.25 pixels
    density_data = grouped_data.join(
        klasse_dist_renamed,
        left_on="naturskog_value",
        right_on="Klasse",
        how="left",
    ).with_columns(
        (pl.col("count") / pl.col("Pixel_Count") * (1_000_000 / (16 * 16))).alias(
            "density_per_km2"
        )
    )
    return (density_data,)


@app.cell
def _(density_data, grouped_data):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Define the desired order of red list categories
    redlist_order = [
        "CR",
        "EN",
        "VU",
        "NT",
        "DD",
        "RE",
    ]

    # Define the desired order for naturskog_value (x-axis)
    naturskog_order = ["Skog", "2-N", "3-N", "4-N", "5-N", "6-N", "7-N"]

    # Prepare count matrix with specific order
    count_matrix = (
        grouped_data.to_pandas()
        .pivot(index="Kategori 2021", columns="naturskog_value", values="count")
        .fillna(0)
    )

    # Reindex to the desired order (rows and columns)
    count_matrix = count_matrix.reindex(
        index=[cat for cat in redlist_order if cat in count_matrix.index],
        columns=[col for col in naturskog_order if col in count_matrix.columns],
    )

    # Prepare density matrix with same order
    density_matrix = (
        density_data.to_pandas()
        .pivot(
            index="Kategori 2021",
            columns="naturskog_value",
            values="density_per_km2",
        )
        .fillna(0)
    )

    density_matrix = density_matrix.reindex(
        index=[cat for cat in redlist_order if cat in density_matrix.index],
        columns=[col for col in naturskog_order if col in density_matrix.columns],
    )

    # Create figure with two subplots stacked vertically
    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Top heatmap: Absolute counts
    sns.heatmap(
        count_matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Antall observasjoner"},
        ax=_ax1,
    )
    _ax1.set_title("Antall observasjoner (75 113)", fontsize=13, pad=15)
    _ax1.set_xlabel("Naturskogsnærhet", fontsize=11)
    _ax1.set_ylabel("Rødlistekategori", fontsize=11)

    # Bottom heatmap: Normalized density
    sns.heatmap(
        density_matrix,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Observasjoner per km²"},
        ax=_ax2,
    )
    _ax2.set_title("Tetthet av observasjoner pr. km² ", fontsize=13, pad=15)
    _ax2.set_xlabel("Naturskogsnærhet", fontsize=11)
    _ax2.set_ylabel("Rødlistekategori", fontsize=11)

    plt.suptitle(
        "Rødlista sopp, lav, moser og karplanter for øvrig natur i skog\n fordelt på vanlig skog og naturskogsklasser (N2-7) ",
        fontsize=14,
        y=1,
    )
    _fig.text(
        0.1,
        0.01,
        "*Analysen omfatter alle artsobservasjoner registrert i skog iht. grunnkart for arealregnskap i hele Norge fra 2010 hvor\n observasjonen er utenfor kartlagte MI-typer, men innenfor dekningsområdet til MI-kartleggingen.",
        ha="left",
        fontsize=10,
        color="black",
        wrap=True,
        verticalalignment="bottom",
    )

    plt.tight_layout(rect=[0.1, 0.05, 1, 0.99], h_pad=3.0)

    plt.gca()
    return plt, sns


@app.cell
def _(density_data, np, plt, sns):
    # Prepare density matrix
    _dens_mat = (
        density_data.to_pandas()
        .pivot(
            index="Kategori 2021",
            columns="naturskog_value",
            values="density_per_km2",
        )
        .fillna(0)
    )

    _rl_order = ["CR", "EN", "VU", "NT", "DD", "RE"]
    _dens_mat = _dens_mat.reindex(
        [cat for cat in _rl_order if cat in _dens_mat.index]
    )

    # Calculate fold change (ratio) from baseline (Skog)
    _fold_change = _dens_mat.copy()
    for col in _fold_change.columns:
        if col != "Skog":
            # Add small constant to avoid division by zero
            _fold_change[col] = (_dens_mat[col] + 1e-10) / (
                _dens_mat["Skog"] + 1e-10
            )

    _fold_change = _fold_change.drop(columns=["Skog"])

    # Create custom annotations showing fold change
    _annot_fc = np.empty(_fold_change.shape, dtype=object)
    for i in range(_fold_change.shape[0]):
        for j in range(_fold_change.shape[1]):
            val = _fold_change.iloc[i, j]
            _annot_fc[i, j] = f"{val:.2f}x"

    _fig_fc, _ax_fc = plt.subplots(1, 1, figsize=(10, 6))

    sns.heatmap(
        _fold_change,
        annot=_annot_fc,
        fmt="",
        cmap="RdYlGn",
        center=1,  # Center at 1x (no change)
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Fold endring (ratio)"},
        ax=_ax_fc,
    )

    _ax_fc.set_title(
        '"Fold change" i observasjonstetthet sammenlignet med skog uten naturskogsnærhet for\nrødlista sopp og lav',
        fontsize=13,
        pad=15,
    )
    _ax_fc.set_xlabel("Naturskogsnærhet", fontsize=11)
    _ax_fc.set_ylabel("Rødlistekategori", fontsize=11)

    _fig_fc.text(
        0.1,
        0.01,
        "*Verdier > 1.0 indikerer høyere tetthet enn i skog uten naturskogsnærhet. 2.0x = dobbelt så høy tetthet.",
        ha="left",
        fontsize=10,
        color="black",
        wrap=True,
        verticalalignment="bottom",
    )

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    plt.gca()
    return


@app.cell(column=3)
def _(mo):
    mo.md(r"""
    ###Forskjell mellom øvrig natur og MI-typer
    """)
    return


@app.cell
def _(arts_obs_for_duckdb, mo, omr_for_duckdb):
    MI_arter_df = mo.sql(
        f"""
        WITH points_in_omr AS (
            SELECT DISTINCT a.*
            FROM arts_obs_for_duckdb a, omr_for_duckdb o
            WHERE ST_Within(
                ST_GeomFromText(a.geometry_wkt), 
                ST_GeomFromText(o.geometry_wkt)
            )
        )
        SELECT * FROM points_in_omr;
        """
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=4)
def _():
    return


if __name__ == "__main__":
    app.run()
