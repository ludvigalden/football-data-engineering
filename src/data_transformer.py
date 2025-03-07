from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from utils import time_execution


@time_execution
def transform_matches_df(matches_df: DataFrame, elo_df: DataFrame) -> DataFrame:
    club_country_mapping = elo_df.select(F.col("Club"), F.col("Country")).distinct()

    # Add Country to matches_df by joining with club_country_mapping
    matches_transformed_df = matches_df.join(
        club_country_mapping, matches_df.HomeTeam == club_country_mapping.Club, "left"
    ).drop("Club")

    # Define Country where null depending on first character of League
    matches_transformed_df = matches_transformed_df.withColumn(
        "Country",
        F.when(
            F.col("Country").isNull(),
            F.when(F.col("League").startswith("EC"), "ENG")
            .when(F.col("League").startswith("SC"), "SCO")
            .when(F.col("League").startswith("SP"), "ESP")
            .when(F.length(F.col("League")) == 3, F.col("League"))
            .when(F.col("League").startswith("E"), "ENG")
            .when(F.col("League").startswith("I"), "ITA")
            .when(F.col("League").startswith("T"), "TUR")
            .when(F.col("League").startswith("F"), "FRA")
            .when(F.col("League").startswith("N"), "NLD")
            .when(F.col("League").startswith("D"), "GER")
            .when(F.col("League").startswith("P"), "PRT")
            .when(F.col("League").startswith("B"), "BEL")
            .when(F.col("League").startswith("G"), "GRE")
            .otherwise("Other"),
        ).otherwise(F.col("Country")),
    )

    # Define unique ID column
    matches_transformed_df = matches_transformed_df.withColumn(
        "ID", F.concat(F.col("Date"), F.lit("_"), F.col("HomeTeam"), F.lit("_"), F.col("AwayTeam"))
    )

    # Define Form3Home, Form5Home, Form3Away, and Form5Away to zero where null
    matches_transformed_df = matches_transformed_df.fillna(
        {
            "Form3Home": 0,
            "Form5Home": 0,
            "Form3Away": 0,
            "Form5Away": 0,
        }
    )

    # When Elo is null, fill with the most recent Elo value for that team
    elo_df_renamed = (
        elo_df.withColumnRenamed("Date", "EloDate")
        .withColumnRenamed("Country", "EloCountry")
        .withColumnRenamed("Club", "EloClub")
    )
    # First, get the latest Elo value for each team before the match date
    latest_home_elo = (
        matches_transformed_df.join(
            elo_df_renamed.alias("home_elo"),
            (matches_transformed_df.HomeTeam == F.col("home_elo.EloClub"))
            & (matches_transformed_df.Date >= F.col("home_elo.EloDate")),
            "left",
        )
        .groupBy("ID")
        .agg(F.max("home_elo.Elo").alias("LatestHomeElo"))
    )

    latest_away_elo = (
        matches_transformed_df.join(
            elo_df_renamed.alias("away_elo"),
            (matches_transformed_df.AwayTeam == F.col("away_elo.EloClub"))
            & (matches_transformed_df.Date >= F.col("away_elo.EloDate")),
            "left",
        )
        .groupBy("ID")
        .agg(F.max("away_elo.Elo").alias("LatestAwayElo"))
    )

    # Join back to original dataframe and update null Elo values only
    matches_transformed_df = matches_transformed_df.join(latest_home_elo, "ID", "left")
    matches_transformed_df = matches_transformed_df.join(latest_away_elo, "ID", "left")

    # Update HomeElo and AwayElo only where they're null
    matches_transformed_df = matches_transformed_df.withColumn(
        "HomeElo", F.when(F.col("HomeElo").isNull(), F.col("LatestHomeElo")).otherwise(F.col("HomeElo"))
    ).withColumn("AwayElo", F.when(F.col("AwayElo").isNull(), F.col("LatestAwayElo")).otherwise(F.col("AwayElo")))

    # Drop all rows where HomeElo AND AwayElo are still null
    matches_transformed_df = matches_transformed_df.dropna(how="any", subset=["HomeElo", "AwayElo"])

    # Drop the temporary columns
    matches_transformed_df = matches_transformed_df.drop("LatestHomeElo", "LatestAwayElo")

    # Calculate EloDiff with null handling
    matches_transformed_df = matches_transformed_df.withColumn(
        "EloDiff",
        F.when(
            (F.col("HomeElo").isNotNull()) & (F.col("AwayElo").isNotNull()),
            F.round(F.col("HomeElo") - F.col("AwayElo"), 2),
        ).otherwise(None),
    )

    # Drop rows where FTHome, FTAway, or FTResult is null
    matches_transformed_df = matches_transformed_df.dropna(how="all", subset=["FTHome", "FTAway", "FTResult"])

    # Make Country, League, ID, MatchTime, HomeTeam, and AwayTeam come first, but keep order of other columns
    matches_transformed_df = matches_transformed_df.select(
        "Country",
        "League",
        "Date",
        "MatchTime",
        "HomeTeam",
        "AwayTeam",
        "EloDiff",
        "FTResult",
        "FTHome",
        "FTAway",
        *[
            column
            for column in matches_transformed_df.columns
            if column
            not in [
                "EloDiff",
                "Country",
                "League",
                "Date",
                "FTResult",
                "FTHome",
                "FTAway",
                "MatchTime",
                "HomeTeam",
                "AwayTeam",
            ]
        ],
    )

    # Order rows by Date, League, MatchTime, and HomeTeam
    matches_transformed_df = matches_transformed_df.orderBy("Date", "League", "MatchTime", "HomeTeam")

    # Drop MatchTime - undefined in half of the rows
    matches_transformed_df = matches_transformed_df.drop("MatchTime")

    # Drop superflous betting odds
    matches_transformed_df = matches_transformed_df.drop("MaxOver25", "MaxUnder25")

    # Drop superflous match info which we will not predict
    matches_transformed_df = matches_transformed_df.drop(
        "HomeShots",
        "AwayShots",
        "HomeFouls",
        "AwayFouls",
        "HomeCorners",
        "AwayCorners",
        "HomeYellow",
        "AwayYellow",
        "HomeRed",
        "AwayRed",
        "HomeRed",
        "AwayRed",
        "HTHome",
        "HTAway",
        "HTResult",
    )

    return matches_transformed_df
