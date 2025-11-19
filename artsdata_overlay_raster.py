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
    return


app._unparsable_cell(
    r"""
    import rasterio
    from pyproj import Transformer

    # Define raster paths
    raster_paths = {
        \"naturskog_value\": \"C:/Users/havh/Downloads/naturskog_v1_naturskognaerhet.tif\",
        \"hogst_flybilde\": \"C:/Users/havh/Downloads/naturskog_v1_støttelag_hogst_flybilde.tif\",
        \"hogst_satelitt\": \"C:/Users/havh/Downloads/naturskog_v1_støttelag_hogst_satelitt.tif\",
    }

    # Dictionary to store values
        raster_values = {}

        for column_name, raster_path in raster_paths.items():
            with rasterio.open(raster_path) as src:
                # PRINT HERE - Check every single map
                print(f\"Processing {column_name}: CRS is {src.crs}\")
            
                # Create transformer for this SPECIFIC raster's CRS
                transformer = Transformer.from_crs(
                    \"EPSG:4326\", src.crs, always_xy=True
                )

            # Transform coordinates
            x_coords, y_coords = transformer.transform(
                arts_obs[\"decimallongitude\"].to_numpy(),
                arts_obs[\"decimallatitude\"].to_numpy(),
            )

            # Sample raster at points
            coords = list(zip(x_coords, y_coords))
            values = [val[0] for val in src.sample(coords)]

            # Store values
            raster_values[column_name] = values

    # Add all columns to dataframe
    arts_obs_with_values = arts_obs.with_columns(
        [pl.Series(name, values) for name, values in raster_values.items()]
    )

    arts_obs_with_values
    """,
    name="_"
)


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
def _(arts_obs_for_duckdb, dekning_for_duckdb, mo, omr_for_duckdb):
    alle_verdier_arter_df = mo.sql(
        f"""
        INSTALL spatial;
        LOAD spatial;

        WITH points_in_omr AS (
            SELECT DISTINCT a.decimallongitude, a.decimallatitude
            FROM arts_obs_for_duckdb a, omr_for_duckdb o
            WHERE ST_Within(
                ST_GeomFromText(a.geometry_wkt), 
                ST_GeomFromText(o.geometry_wkt)
            )
        ),
        points_in_dekning AS (
            SELECT DISTINCT a.decimallongitude, a.decimallatitude
            FROM arts_obs_for_duckdb a, dekning_for_duckdb d
            WHERE ST_Within(
                ST_GeomFromText(a.geometry_wkt), 
                ST_GeomFromText(d.geometry_wkt)
            )
        )
        SELECT 
            a.*,
            o.decimallongitude IS NOT NULL AS in_omr,
            d.decimallongitude IS NOT NULL AS in_dekning,
            (o.decimallongitude IS NULL AND d.decimallongitude IS NOT NULL) AS outside_omr_inside_dekning
        FROM arts_obs_for_duckdb a
        LEFT JOIN points_in_omr o 
            ON a.decimallongitude = o.decimallongitude 
            AND a.decimallatitude = o.decimallatitude
        LEFT JOIN points_in_dekning d 
            ON a.decimallongitude = d.decimallongitude 
            AND a.decimallatitude = d.decimallatitude;
        """
    )
    return (alle_verdier_arter_df,)


@app.cell(column=2)
def _(alle_verdier_arter_df, mo):
    arter_df = mo.ui.table(alle_verdier_arter_df)
    arter_df
    return (arter_df,)


@app.cell
def _(arter_df, pl):
    combined_data = (
        arter_df.value.group_by("Kategori 2021")
        .agg(
            [
                pl.col("in_omr")
                .sum()
                .alias(
                    "MI-typer"
                ),  # true=1 og false=0, slik at sum er da lik alle som møter dette kriteriet.
                pl.col("outside_omr_inside_dekning")
                .sum()
                .alias("Kartlagt øvrig natur - skog"),
                pl.col("naturskog_value")
                .is_between(2, 7)
                .sum()
                .alias("Naturskogsnærhet 2-7"),
                (pl.col("naturskog_value") == 1).sum().alias("Naturskogsnærhet 1"),
            ]
        )
        .unpivot(
            index="Kategori 2021", variable_name="category", value_name="count"
        )
    )

    combined_data
    return (combined_data,)


@app.cell
def _(combined_data, plt, redlist_order, sns):
    # Create pivot table for heatmap
    pivot_data = (
        combined_data.to_pandas()
        .pivot(index="category", columns="Kategori 2021", values="count")
        .fillna(0)
    )

    # Define desired order for categories (y-axis)
    category_order = [
        "MI-typer",
        "Kartlagt øvrig natur - skog",
        "Naturskogsnærhet 2-7",
        "Naturskogsnærhet 1",
    ]


    # Reindex to desired order
    pivot_data = pivot_data.reindex(
        index=[cat for cat in category_order if cat in pivot_data.index],
        columns=[col for col in redlist_order if col in pivot_data.columns],
    )

    # Add sum column
    pivot_data["Sum"] = pivot_data.sum(axis=1)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(11, 6))

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Antall observasjoner"},
        ax=ax,
    )

    # Replace comma separators with spaces
    for text in ax.texts:
        value = text.get_text()
        # Format with thousand separator and replace comma with space
        text.set_text(f"{float(value):,.0f}".replace(",", " "))

    ax.set_title(
        "Fordeling av rødlista lav, sopp og moser registrert i skog etter 2010 på ulike kategorier",
        fontsize=13,
        pad=15,
    )
    ax.set_xlabel("Rødlistekategori", fontsize=11)
    ax.set_ylabel("", fontsize=11)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(column=3)
def _(mo):
    mo.md(r"""
    ### Husk at tetthet kun er for nasjonale data, du har ikke for lokale. Webjørn har lagd, men ikke med klasse 1. Usikker på om du skal be han gjøre det?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Statestikk på fordeling av arter i ulike anturskogsnærhetklasser
    Velg artsutvalg og "raster utvalg"
    """)
    return


@app.cell(hide_code=True)
def _(arter_filtrert_MI_df, arts_obs_with_values, mo):
    dropdown = mo.ui.dropdown(
        options={
            "Alle arter": arts_obs_with_values,
            "Arter innad dekning og utad MI-type": arter_filtrert_MI_df,
        },
        label="Velg artsutvalg",
        value="Arter innad dekning og utad MI-type",
    )
    dropdown
    return (dropdown,)


@app.cell
def _(dropdown, mo):
    # Brukke denne til å filtrer i figuren
    arter_riktig_df = mo.ui.table(dropdown.value, selection="multi")
    arter_riktig_df
    return (arter_riktig_df,)


@app.cell(hide_code=True)
def _(density_data, filtered_data, grouped_data):
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
    naturskog_order = ["1-N", "2-N", "3-N", "4-N", "5-N", "6-N", "7-N"]

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
    _ax1.set_title(
        f"Antall observasjoner ({len(filtered_data)})", fontsize=13, pad=15
    )
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
        "Rødlista sopp, lav, moser og karplanter for øvrig natur i skog\n fordelt på naturskogsklasser (N1-7) ",
        fontsize=14,
        y=1,
    )
    _fig.text(
        0.1,
        0.01,
        "*Analysen omfatter alle artsobservasjoner registrert i skog iht. SR-16 avgrensninger i hele Norge fra 2010.",
        ha="left",
        fontsize=10,
        color="black",
        wrap=True,
        verticalalignment="bottom",
    )

    plt.tight_layout(rect=[0.1, 0.05, 1, 0.99], h_pad=3.0)

    plt.gca()
    return plt, redlist_order, sns


@app.cell
def _(naturskog, np, pl):
    unike_klasser_nasjonalt, counts = np.unique(
        naturskog.values.flatten(), return_counts=True
    )
    klasse_distribusjon = pl.DataFrame(
        {"Klasse": unike_klasser_nasjonalt, "Pixel_Count": counts}
    )

    klasse_distribusjon
    return (klasse_distribusjon,)


@app.cell
def _():
    mapping = {
        255: "Ikke skog",
        1: "1-N",
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
    filtered_data = arter_riktig_df.value

    grouped_data_med_ikke_skog = (
        filtered_data.filter(~pl.col("Kategori 2021").is_in(["DD", "NT°"]))
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
    return filtered_data, grouped_data_med_ikke_skog


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
def _(mo):
    mo.md(r"""
    ### Analyse av hogstår sett opp mot rødlistearter
    """)
    return


@app.cell
def _(arts_obs_with_values, pl):
    # Bruk arts_obs_with_values
    arter_med_hogsttidspunkt = arts_obs_with_values.filter(
        pl.col("hogst_satelitt") != 0
    )
    arter_med_hogsttidspunkt
    return


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
def _():
    return


@app.cell
def _(arts_obs_for_duckdb, dekning_for_duckdb, mo, omr_for_duckdb):
    arter_filtrert_MI_df = mo.sql(
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
    return (arter_filtrert_MI_df,)


@app.cell(column=4)
def _():
    return


if __name__ == "__main__":
    app.run()
