import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from typing import List
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
        ]
    )

    print("Rating Matrix:")
    print(ratings)
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
    mo.md(r"""# Mean Centered Ratings""")
    return


@app.cell
def _(mean_ratings, ratings):
    mean_centered = ratings - mean_ratings.reshape(-1, 1)
    print("Mean Centered Ratings")
    print(mean_centered)
    return (mean_centered,)


@app.cell
def _(mo):
    mo.md(r"""# Calculate Adjusted Cosine""")
    return


@app.cell
def _(List, np):
    def calc_adj_cosine(s_i: np.ndarray | List, s_j: np.ndarray | List) -> float:
        """
        Calculates the adjusted cosine between two mean-centered vectors.

        Parameters:
        -----------
        s_i: np.ndarray
            Mean-centered ratings of item 'i'.
        s_j: np.ndarray
            Mean-centered ratings of item 'j'.

        Returns:
        --------
        Adjusted cosine similarity between items 'i' and 'j'
        """

        if not isinstance(s_i, np.ndarray):
            s_i = np.asarray(s_i)

        if not isinstance(s_j, np.ndarray):
            s_j = np.asarray(s_j)

        users_rated_i = np.where(~np.isnan(s_i))[0]
        users_rated_j = np.where(~np.isnan(s_j))[0]
        common_users = np.intersect1d(users_rated_i, users_rated_j).tolist()

        if len(common_users) == 0:
            return 0

        s_i = s_i[common_users]
        s_j = s_j[common_users]

        num = np.sum(s_i * s_j)
        denom = np.sqrt(np.sum(s_i**2)) * np.sqrt(np.sum(s_j**2))
        return round(num / denom, 3)
    return (calc_adj_cosine,)


@app.cell
def _(calc_adj_cosine, np):
    def calc_similarities(
        target_col: int, mean_centered_ratings: np.ndarray
    ) -> np.ndarray:
        """
        Calculates similarities of other items to target item.

        Parameters:
        -----------
        target_col: int
            The target item.

        mean_centered_ratings: np.ndarray
            The mean-centered ratings matrix.

        Returns:
        ---------
        Similarities of the target item to the other items.
        """

        if target_col >= mean_centered_ratings.shape[1] or target_col < 0:
            raise ValueError(
                f"Target column has be in the range 0-{(mean_centered_ratings.shape[1] - 1)}. Found {target_col}"
            )

        target_vec = mean_centered_ratings[:, target_col]
        similarities = list()
        for i in range(mean_centered_ratings.shape[1]):
            similarities.append(
                calc_adj_cosine(s_i=mean_centered_ratings[:, i], s_j=target_vec)
            )

        return np.asarray(similarities)
    return (calc_similarities,)


@app.cell
def _(calc_similarities, mean_centered):
    sim_item_1 = calc_similarities(
        target_col=0, mean_centered_ratings=mean_centered
    )
    sim_item_6 = calc_similarities(
        target_col=5, mean_centered_ratings=mean_centered
    )

    print(f"Similarities of items to item-1: {sim_item_1}")
    print(f"Similarities of items to item-6: {sim_item_6}")
    return sim_item_1, sim_item_6


@app.cell
def _(mo):
    mo.md(r"""# Calculate Predictions""")
    return


@app.cell
def _(mean_centered_ratings, np):
    def predict(
        similarities: np.ndarray,
        ratings_mat: np.ndarray,
        target_user: int,
        top_k: int = 2,
    ) -> float:
        """
        Predicts the ratings for a target item of a given user.

        Parameters:
        ------------
        similarities: np.ndarray
            Similarities of the target item to all other items.

        ratings_mat: np.ndarray
            The complete rating matrix.

        target_user: int
            The target user.

        top_k: int, default=2
            The top-k items to fetch.

        Returns:
        ------------
        The predicted rating of the target item for a given user.
        """
        assert isinstance(ratings_mat, np.ndarray), (
            f"Expected ratings_mat to be of type np.ndarray. Found {type(ratings_mat)}"
        )

        if target_user >= ratings_mat.shape[0] or target_user < 0:
            raise ValueError(
                f"Target User has be in the range 0-{(mean_centered_ratings.shape[0] - 1)}. Found {target_user}"
            )

        target_ratings = ratings_mat[target_user, :]

        if top_k >= ratings_mat.shape[0]:
            top_k = ratings_mat.shape[0]

        items_user_rated = np.where(~np.isnan(target_ratings))[0]
        target_ratings = target_ratings[items_user_rated]
        similarities = similarities[items_user_rated]

        sorted_idxs = np.argsort(similarities)[::-1]
        top_k_items = sorted_idxs[:top_k].tolist()
        num = np.sum(similarities[top_k_items] * target_ratings[top_k_items])
        denom = np.sum(np.abs(similarities[top_k_items]))

        return round(num / denom, 3)
    return (predict,)


@app.cell
def _(predict, ratings, sim_item_1, sim_item_6):
    r_item1 = predict(similarities=sim_item_1, ratings_mat=ratings, target_user=2)
    r_item6 = predict(similarities=sim_item_6, ratings_mat=ratings, target_user=2)

    print(f"Predicted Rating for item-1: {r_item1}")
    print(f"Predicted Rating for item-6: {r_item6}")
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


if __name__ == "__main__":
    app.run()
