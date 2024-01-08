import copy

import numpy as np
import pandas as pd
import tqdm

from .base_165_features import *


def get_all_features(df):
    """
    Get all features for the given dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The dataframe with all the features.
    """
    # feature_df = get_paragraph_features(df)
    # return feature_df

    # NOTE: ugly hack to utilize base_165_features without using anything else
    DATA_ROOT = (
        "/Users/kaiqu/kaggle-datasets/linking-writing-processes-to-writing-quality"
    )
    train_logs = pl.scan_csv(f"{DATA_ROOT}/train_logs.csv")
    feature_extractor = FeatureExtractor(train_logs)
    train_features = feature_extractor.create_feats().collect().to_pandas()
    return train_features


def get_word_features(df, essay_df=None):
    # TODO: input word average length + std
    # TODO:
    pass


def get_sentence_features(df, essay_df=None):
    pass


# ? Do we have to make sure that the passed df are not changed in anyway?
# ? What actions of df will return a new df?
def get_paragraph_features(df, essay_df=None):
    # feature_df = df[["id"]].copy()
    # get the word length features
    # TODO: need to rewrite the below shit; figure out a more elegant way to make sure that the indices are aligned
    df_max_event = df.loc[df.groupby("id")["event_id"].idxmax()]
    # print("max event", df_max_event.shape)
    # feature_df = df_max_event.merge(feature_df, on="id", how="left")
    # print("feature_df", feature_df.shape)
    feature_df = df_max_event[["id", "word_count"]].reset_index(drop=True)
    return feature_df


def get_other_features(df, essay_df=None):
    # TODO: time spent on the essay
    # TODO: count activities features
    # TODO: count events features
    # TODO: action time mean, std, median features
    # ? text_changes features
    pass


def get_behavioral_features(df, essay_df=None):
    # TODO: number of P-bursts (in total or per minute)
    # TODO: number of R-bursts (in total or per minute)
    # TODO: proportion of P-bursts (as a % of total writing time)
    # TODO: proportion of R-bursts (as a % of total writing time)
    # TODO: length of P-bursts (in characters)
    # TODO: length of R-bursts (in characters)
    pass

def get_knn_features(df):


# def preprocess_df(df):
#     pass


class EssayConstructor:
    def preprocess_df(self, df):
        essay_text = ""
        # input[0] is activity, input[1] is cursor_position, input[2] is text_change, input[3] is id
        for input in df.values:
            activity, cursor_position, text_change = input[0], input[1], input[2]
            # exmaple of replacing the text: qqqqqqqqq qq  => q
            if activity == "Replace":
                # splits text_change at ' => '
                text_change_splits = text_change.split(" => ")
                text_before, text_after = text_change_splits[0], text_change_splits[1]
                essay_text = (
                    essay_text[: cursor_position - len(text_after)]
                    + text_after
                    + essay_text[cursor_position - len(text_after) + len(text_before) :]
                )
            elif activity == "Paste":
                essay_text = (
                    essay_text[: cursor_position - len(text_change)]
                    + text_change
                    + essay_text[cursor_position - len(text_change) :]
                )
            elif activity == "Remove/Cut":
                essay_text = (
                    essay_text[:cursor_position]
                    + essay_text[cursor_position + len(text_change) :]
                )
            # if activity is Move. e.g., 'Move From [287, 289] To [285, 287]'
            elif activity.startswith("Move"):
                # get rid of the "Move from " text
                cropped_txt = activity[10:]
                # splits cropped text by ' To '
                split_txt = cropped_txt.split(" To ")
                # splits split text again by ', ' for each item
                move_pos = [item.split(", ") for item in split_txt]
                # move from [2, 4] To [5, 7] = (2, 4, 5, 7)
                original_row, original_col, new_row, new_col = (
                    int(move_pos[0][0][1:]),  # get rid of '['
                    int(move_pos[0][1][:-1]),  # get rid of ']'
                    int(move_pos[1][0][1:]),
                    int(move_pos[1][1][:-1]),
                )
                # Skip if someone manages to activiate this by moving to same place
                if original_row != new_row:
                    # Check if they move text forward in essay (they are different)
                    if original_row < new_row:
                        essay_text = (
                            essay_text[:original_row]
                            + essay_text[original_col:new_col]
                            + essay_text[original_row:original_col]
                            + essay_text[new_col:]
                        )
                    else:
                        essay_text = (
                            essay_text[:new_row]
                            + essay_text[original_row:original_col]
                            + essay_text[new_row:original_row]
                            + essay_text[original_col:]
                        )
            else:
                # If activity is Input
                essay_text = (
                    essay_text[: cursor_position - len(text_change)]
                    + text_change
                    + essay_text[cursor_position - len(text_change) :]
                )
        return essay_text

    def get_essay_df(self, df):
        text_df = df[["id", "activity", "cursor_position", "text_change"]].copy()
        # get rid of text inputs that make no change
        text_df = text_df[text_df.activity != "Nonproduction"]
        # construct essay using the fast way
        tqdm.pandas()
        essay = text_df.groupby("id")[
            ["activity", "cursor_position", "text_change"]
        ].progress_apply(lambda x: self.preprocess_df(x))
        essay_df = essay.to_frame().reset_index()
        essay_df.columns = ["id", "essay"]
        return essay_df


if __name__ == "__main__":
    print("TODO")
