import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(np):
    ratings_mat = np.array(
        [
            [7, 6, 7, 4, 5, 4],
            [6, 7, np.nan, 4, 3, 4],
            [np.nan, 3, 3, 1, 1, np.nan],
            [1, 2, 2, 3, 3, 4],
            [1, np.nan, 1, 2, 3, 3],
        ]
    )
    return (ratings_mat,)


@app.cell
def _(np):
    def pearson_coeff(r_u: np.ndarray, r_v: np.ndarray, mu_u: float, mu_v: float) -> float:
        """
        Calculates the Pearson correlation coefficient between two vectors

        Parameters:
        ----------
        r_u : np.ndarray
            Ratings of user u
        r_v : np.ndarray
            Ratings of user v

        mu_u: float
            mean rating of user u

        mu_v: float
            mean rating of user v

        Returns:
        --------
        Pearson correlation coefficient between user 'u' and 'v'
        """

        u_rated_items = np.where(~np.isnan(r_u))[0]
        v_rated_items = np.where(~np.isnan(r_v))[0]

        common_rated_items = np.intersect1d(u_rated_items, v_rated_items).tolist()

        r_u = r_u[common_rated_items]
        r_v = r_v[common_rated_items]

        a = r_u - mu_u
        b = r_v - mu_v

        num = np.sum(a * b)
        denom = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))

        return round(num / denom, 3)
    return (pearson_coeff,)


@app.cell
def _(np):
    def cosine_sim(r_u: np.ndarray, r_v: np.ndarray) -> float:
        """
        Calculates the cosine similarity between two vectors

        Parameters:
        ----------
        r_u : np.ndarray
            Ratings of user u

        r_v : np.ndarray
            Ratings of user v

        Returns:
        --------
        Cosine similarity between user 'u' and 'v'
        """

        u_rated_items = np.where(~np.isnan(r_u))[0]
        v_rated_items = np.where(~np.isnan(r_v))[0]

        common_rated_items = np.intersect1d(u_rated_items, v_rated_items).tolist()

        r_u = r_u[common_rated_items]
        r_v = r_v[common_rated_items]

        num = np.sum(r_u * r_v)
        denom = np.sqrt(np.sum(r_u**2)) * np.sqrt(np.sum(r_v**2))

        return round(num / denom, 3)
    return (cosine_sim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Calculate Mean Ratings""")
    return


@app.cell
def _(np, ratings_mat):
    mean_ratings = np.nanmean(ratings_mat, axis=1)
    mean_ratings
    return (mean_ratings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Calculate User Similarity to a target user""")
    return


@app.cell
def _(cosine_sim, mean_ratings, pearson_coeff, ratings_mat):
    sim_pearson = list()
    sim_cosine = list()

    ratings_user_3 = ratings_mat[2]  # Target user

    for i in range(ratings_mat.shape[0]):
        if i == 2:
            continue

        sim_pearson.append(
            pearson_coeff(ratings_mat[i], ratings_user_3, mean_ratings[i], mean_ratings[2])
        )
        sim_cosine.append(cosine_sim(ratings_mat[i], ratings_user_3))


    print(f"Pearson Correlation Similarities to user 3: {sim_pearson}")
    print(f"Cosine Similarities to user 3: {sim_cosine}")
    return (sim_pearson,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Calculate Mean-Centered Ratings""")
    return


@app.cell
def _(mean_ratings, ratings_mat):
    mean_centered_ratings_mat = ratings_mat - mean_ratings.reshape(-1, 1)
    mean_centered_ratings_mat
    return (mean_centered_ratings_mat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Calculate Predictions for item 1 and 6 of user 3""")
    return


@app.cell
def _(mean_centered_ratings_mat, mean_ratings, np, sim_pearson):
    # Since the most similar users to user 3 are user 1 and user 2
    # as per the Pearson Correlation Coefficient Similarity.

    r_v = mean_centered_ratings_mat[[0, 1]]
    mu_v = mean_ratings[[0, 1]]
    sim = np.asarray(sim_pearson)[[0, 1]]
    mu_u = mean_ratings[2]


    r_3_0 = mu_u + np.sum(sim * r_v[[0, 1], 0]) / np.sum(np.abs(sim))
    r_3_6 = mu_u + np.sum(sim * r_v[[0, 1], 5]) / np.sum(np.abs(sim))

    print(f"Predicted Rating for item - 1: {r_3_0:.2f}")
    print(f"Predicted Rating for item - 6: {r_3_6:.2f}")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
