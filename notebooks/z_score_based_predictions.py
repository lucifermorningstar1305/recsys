import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from typing import List
    import marimo as mo
    import numpy as np
    return List, mo, np


@app.cell
def _(np):
    ratings = np.array(
        [
            [7, 6, 7, 4, 5, 4],
            [6, 7, np.nan, 4, 3, 4],
            [np.nan, 3, 3, 1, 1, np.nan],
            [1, 2, 2, 3, 3, 4],
            [1, np.nan, 1, 2, 3, 3],
            [np.nan, np.nan, np.nan, 6, 1, 2],
            [1, 1, 2, 2, np.nan, 7],
            [8, 8, np.nan, np.nan, 4, 2],
        ]
    )
    return (ratings,)


@app.cell
def _(mo):
    mo.md(r"""# Calculate Mean Ratings""")
    return


@app.cell
def _(np, ratings):
    mean_ratings = np.nanmean(ratings, axis=1)
    print(f"Mean Ratings: {mean_ratings}")
    return (mean_ratings,)


@app.cell
def _(mo):
    mo.md(r"""# Calculate User Similarities""")
    return


@app.cell
def _(np):
    def calc_pearson_sim(
        r_u: np.ndarray, r_v: np.ndarray, mu_u: float, mu_v: float, beta: int = 2
    ) -> float:
        """
        Calculates the Pearson correlation coefficient between two vectors.

        Paramters:
        ----------
        r_u: np.ndarray
            Ratings vector of user 'u'
        r_v: np.ndarray
            Ratings vector of user 'v'

        mu_u: float
            Mean rating of user 'u'

        mu_v: float
            Mean rating of user 'v'

        beta: int, default=2
            Minimum number of common items rated by users 'u' and 'v'

        Returns:
        --------
        Pearson correlation coefficient between the ratings of two user 'u' and 'v'
        """

        u_rated_items = np.where(~np.isnan(r_u))[0]
        v_rated_items = np.where(~np.isnan(r_v))[0]

        common_items = np.intersect1d(u_rated_items, v_rated_items).tolist()
        r_u = r_u[common_items]
        r_v = r_v[common_items]

        a = r_u - mu_u
        b = r_v - mu_v

        num = np.sum(a * b)
        denom = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))
        sim = (num / denom) * (min(len(common_items), beta) / beta)
        return sim
    return (calc_pearson_sim,)


@app.cell
def _(calc_pearson_sim, mean_ratings, ratings):
    target_row = 2  # Target user

    similarities = list()

    for i in range(ratings.shape[0]):
        if i == target_row:
            continue

        similarities.append(
            calc_pearson_sim(
                r_u=ratings[i],
                r_v=ratings[target_row],
                mu_u=mean_ratings[i],
                mu_v=mean_ratings[target_row],
            )
        )

    print(f"User similarities to user-3: {similarities}")
    return similarities, target_row


@app.cell
def _(mo):
    mo.md(r"""# Calculate Z-Score based ratings""")
    return


@app.cell
def _(mean_ratings, np, ratings):
    mean_centered = ratings - mean_ratings.reshape(-1, 1)
    deviations = np.sqrt(
        np.nansum(mean_centered**2, axis=1)
        / (np.count_nonzero(~np.isnan(mean_centered), axis=1) - 1)
    )
    z_scored = mean_centered / deviations.reshape(-1, 1)
    z_scored
    return deviations, z_scored


@app.cell
def _(mo):
    mo.md(r"""# Calculate Predictions""")
    return


@app.cell
def _(List, np):
    def predict(
        sims: np.ndarray | List,
        z_scored_ratings: np.ndarray,
        mu: float,
        sigma: float,
        top_k: int = 2,
    ) -> float:
        """
        Predicts ratings for an item.

        Parameters:
        -----------
        sims: np.ndarray | List
            User similarities

        z_scored_ratings: np.ndarray
            Z-scored centered ratings

        mu: float
            Mean rating of the target user

        sigma: float
            Standard deviation of rating of the target user

        top_k: int, default = 2
            Top-k similar users to the current user.
        """

        assert isinstance(z_scored_ratings, np.ndarray), (
            f"Expected z_scored_ratings to be a numpy array. Found {type(z_scored_ratings)}"
        )

        assert top_k >= 1, f"Expected top_k to be >= 1. Found top_k = {top_k}"

        if isinstance(sims, list):
            sims = np.asarray(sims)

        sorted_idxs = np.argsort(sims)[::-1][:top_k].tolist()
        filtered_sims = sims[sorted_idxs]

        z_scored_ratings = z_scored_ratings[sorted_idxs]

        return mu + sigma * (
            np.sum(filtered_sims * z_scored_ratings)
            / np.sum(np.abs(filtered_sims))
        )
    return (predict,)


@app.cell
def _(deviations, mean_ratings, predict, similarities, target_row, z_scored):
    predicted_rating_item_0 = predict(
        sims=similarities,
        z_scored_ratings=z_scored[:, 0],
        mu=mean_ratings[target_row],
        sigma=deviations[target_row],
    )
    predicted_rating_item_5 = predict(
        sims=similarities,
        z_scored_ratings=z_scored[:, 5],
        mu=mean_ratings[target_row],
        sigma=deviations[target_row],
    )

    print(f"Predicted Rating of item-1: {predicted_rating_item_0:.2f}")
    print(f"Predicted Rating of item-6: {predicted_rating_item_5:.2f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
