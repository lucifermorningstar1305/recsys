import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _(mo):
    mo.md("""# Discounted Similarity""")
    return


@app.cell
def _(np):
    ratings = np.array([
        [7, 6, 7, 4, 5, 4],
        [6, 7, np.nan, 4, 3, 4],
        [np.nan, 3, 3, 1, 1, np.nan],
        [1, 2, 2, 3, 3, 4],
        [1, np.nan, 1, 2, 3, 3],
        [np.nan, np.nan, np.nan, 6, 1, 2],
        [1, 1, 2, 2, np.nan, 7]
    ])

    print(ratings.shape)
    return (ratings,)


@app.cell
def _(mo):
    mo.md(r"""# Calculate mean ratings""")
    return


@app.cell
def _(np, ratings):
    mean_ratings = np.nanmean(ratings, axis=1)
    print(f"Mean ratings = {mean_ratings}")
    return (mean_ratings,)


@app.cell
def _(mo):
    mo.md("""# Calculate User Similarities""")
    return


@app.cell
def _(np):
    def calc_pearson_sim(r_u: np.ndarray, r_v: np.ndarray, mu_u: float, mu_v: float, beta: int=2) -> float:
        """
        Calculates the Pearson Correlation Coefficient between two vectors.

        Parameters:
        -----------
        r_u: np.ndarray
            Ratings of user 'u'

        r_v: np.ndarray
            Ratings of user 'v'

        mu_u: float
            Mean rating of user 'u'

        mu_v: float
            Mean rating of user 'v'

        beta: int, default=2
            Threshold for minimum number of common items rated by users 'u' and 'v'

        Returns:
        --------
        The Pearson correlation Coefficient similarity.
        """

        u_rated_items = np.where(~np.isnan(r_u))[0]
        v_rated_items = np.where(~np.isnan(r_v))[0]

        common_items = np.intersect1d(u_rated_items, v_rated_items).tolist()
        r_u = r_u[common_items]
        r_v = r_v[common_items]

        a = r_u - mu_u
        b = r_v - mu_v

        num = np.sum(a * b)
        denom = np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2))
        sim = (num / denom) * (min(len(common_items), beta) / beta)
        return sim

    return (calc_pearson_sim,)


@app.cell
def _(calc_pearson_sim, mean_ratings, ratings):
    target_row = 2 # Target user

    similarities = list()

    for i in range(ratings.shape[0]):
        if i == target_row: continue

        similarities.append(calc_pearson_sim(r_u=ratings[i], r_v=ratings[target_row], mu_u=mean_ratings[i],
                                             mu_v=mean_ratings[target_row]))

    print(f"User similarities to user-3: {similarities}")

    return (similarities,)


@app.cell
def _(mo):
    mo.md("# Calculate Mean Centered Ratings")
    return


@app.cell
def _(mean_ratings, ratings):
    mean_centered_ratings_mat = ratings - mean_ratings.reshape(-1, 1)
    mean_centered_ratings_mat
    return (mean_centered_ratings_mat,)


@app.cell
def _(mo):
    mo.md("# Calculate Predictions")
    return


@app.cell
def _(mean_centered_ratings_mat, mean_ratings, np, similarities):
    # Since the most similar users to user 3 are user 1 and user 2
    # as per the Pearson Correlation Coefficient Similarity.

    r_v = mean_centered_ratings_mat[[0, 1]]
    mu_v = mean_ratings[[0, 1]]
    sim = np.asarray(similarities)[[0, 1]]
    mu_u = mean_ratings[2]


    r_3_0 = mu_u + np.sum(sim * r_v[[0, 1], 0]) / np.sum(np.abs(sim))
    r_3_6 = mu_u + np.sum(sim * r_v[[0, 1], 5]) / np.sum(np.abs(sim))

    print(f"Predicted Rating for item - 1: {r_3_0:.2f}")
    print(f"Predicted Rating for item - 6: {r_3_6:.2f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
