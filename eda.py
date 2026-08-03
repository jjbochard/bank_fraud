# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: venv (3.14.6.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1. Imports

# %%
from collections.abc import Sequence
from typing import Literal, cast

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from pandas import DataFrame

plt.style.use("dark_background")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["figure.constrained_layout.use"] = True

FRAUD_LEGEND = ["Non fraudulent", "Fraudulent"]
type NORMALIZE_TYPE = Literal["index", "columns"]


# %%
def build_graph(
    ax: Axes,
    title: str,
    y_label: str | None = None,
    legend_labels: Sequence[str] | None = None,
) -> None:
    """
    Configure the appearance of a matplotlib chart.

    Set the chart title, optionally updates the y-axis label, and
    customize the legend by removing its title and replacing the legend
    labels.

    Args:
        ax: Matplotlib axes containing the plot.
        title: Chart title.
        y_label: Label for the y-axis. If None, the current label is
            left unchanged.
        legend_labels: Labels used to replace the existing legend labels.
            If None, the legend labels are not modified.
    """

    ax.set_title(title)

    if y_label:
        ax.set_ylabel(y_label)

    legend = ax.get_legend()

    if legend is None:
        return

    legend.set_title("")

    if legend_labels:
        for text, label in zip(legend.get_texts(), legend_labels):
            text.set_text(label)


def analyze_categorical(
    df: DataFrame,
    col: str,
    title: str,
    y_label: str | None,
    legend_labels: Sequence[str] | None,
    index_column_name: str,
    table_1_name: str,
    table_2_name: str,
    table_3_name: str,
    round_format: str = "%.02f%%",
    is_bar_label: bool = True,
) -> None:
    """
    Analyze and visualize a categorical feature.

    Create a count plot showing the percentage distribution of a
    categorical feature by fraud status.
    Also display summary tables containing fraud counts and normalized
    contingency tables.

        Args:
        df: Input dataframe.
        col: Name of the categorical column to analyze.
        title: Plot title.
        y_label: Label for the y-axis.
        legend_labels: Labels used to replace the default legend labels.
        index_column_name: Name of the count column in the first summary table.
        table_1_name: Heading displayed above the fraud-count table.
        table_2_name: Heading displayed above the column-normalized
        contingency table.
        table_3_name: Heading displayed above the row-normalized
        contingency table.
        round_format: Format string used for bar labels.
        is_bar_label: Whether to display values on top of the bars.
    """
    ax = sns.countplot(
        data=df,
        x=col,
        stat="percent",
        hue="is_fraud",
        order=df[col].value_counts().index,
    )
    build_graph(
        ax,
        title,
        y_label,
        legend_labels,
    )
    if is_bar_label:
        for container in ax.containers:
            ax.bar_label(cast(BarContainer, container), fmt=round_format)

    plt.show()

    # Number of merchant category by fraud count
    table_1 = (
        df.astype({"is_fraud": "int"})
        .groupby(col, observed=True)["is_fraud"]
        .sum()
        .value_counts()
        .sort_index()
        .rename_axis("number_of_frauds")
        .reset_index(name=index_column_name)
    )

    display(
        Markdown(f"## {table_1_name}"),
        table_1,
    )

    map_normalize_type_title: dict[NORMALIZE_TYPE, str] = {
        "columns": table_2_name,
        "index": table_3_name,
    }

    for normalize_type, table_title in map_normalize_type_title.items():
        display(
            Markdown(f"## {table_title}"),
            pd.crosstab(
                df[col],
                df["is_fraud"],
                margins=True,
                normalize=normalize_type,
            ),
        )


# %%
df = pd.read_csv("./downloads/fraud_enriched.csv")


# %% [markdown]
# # 2. Data understanding

# %%
df.shape


# %%
df.head()


# %%
df.columns


# %%
df.dtypes


# %%
df.describe().T


# %% [markdown]
# # 3. Data preparation

# %%
df.head()


# %%
df["event_ts"] = pd.to_datetime(df["event_ts"])


# %%
date_parts = {
    "day_event": "day",
    "weekday_event": "weekday",
    "month_event": "month",
    "year_event": "year",
    "hour_event": "hour",
}

dt = df["event_ts"].dt

for column, attribute in date_parts.items():
    df[column] = getattr(dt, attribute)


# %%
df.dtypes


# %%
df.columns = map(str.lower, df.columns)
df = df.rename(columns={"class": "is_fraud"})


# %%
type_mapping = {
    "merchant_category_code": "category",
    "merchant_country": "category",
    "currency": "category",
    "device_type": "category",
    "entry_mode": "category",
    "channel": "category",
    "is_international": "category",
    "is_fraud": "category",
}
df = df.astype(type_mapping)


# %%
df.describe()


# %%
df.describe(include=["category"])


# %%
df.isna().sum()


# %%
display(df.loc[df.duplicated()])

columns_subsets = [
    ["transaction_id"],
    df.columns.to_list()[1:],  # All but transation_id
    df.columns.to_list()[2:30],  # Only starting v columns
    df.columns.to_list()[30:],  # All but transaction_id, time and starting v columns
]
for s in columns_subsets:
    if df.loc[df.duplicated(subset=s)].empty is True:
        print("Subset has no duplicated values")
    else:
        print(f"Check {s} subset")


# %%
df.dtypes
display(df.head(5))


# %% [markdown]
# # 4. Feature understanding

# %% [markdown]
# ## Univariate analysis

# %% [markdown]
# ### Is_fraud

# %%
fig, ax = plt.subplots(layout="constrained")

sns.countplot(data=df, x="is_fraud", stat="percent", ax=ax)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Not fraud", "Fraud"])

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%")

ax.set_ylabel("Percentage")
ax.set_title("Distribution of Fraudulent and Non-Fraudulent Transactions")
plt.show()


# %% [markdown]
# ### time

# %%
col_to_plot = "time"

fig = sns.histplot(
    df,
    x=col_to_plot,
    stat="density",
    hue="is_fraud",
    kde=True,
    bins=15,
    common_norm=False,
    legend=True,
)

fig.set_title(f"Distribution of transactions {col_to_plot}")
fig.set_xlabel(f"{col_to_plot.capitalize()}")
fig.set_ylabel("Density")
plt.show()


# %% [markdown]
# ### V columns

# %%
# Choose a range or a list of numbers between 1 and 28
# in order to not display all plots.

cols = range(1, 4)

for col in cols:
    fig = sns.histplot(
        df,
        x=f"v{col}",
        stat="density",
        hue="is_fraud",
        kde=True,
        # bins=15,
        common_norm=False,
        legend=True,
    )

    fig.set_title(f"Distribution of v{col}")
    fig.set_xlabel(f"v{col}")
    fig.set_ylabel("Density")
    plt.show()


# %% [markdown]
# ### Amount

# %%
col_to_plot = "amount"

fig = sns.histplot(
    df,
    x=col_to_plot,
    stat="density",
    hue="is_fraud",
    kde=True,
    common_norm=False,
    legend=True,
)

fig.set_xlim(0, 250)
fig.set_title(f"Distribution of transactions {col_to_plot} lees than 100")
fig.set_xlabel(f"{col_to_plot.capitalize()}")
fig.set_ylabel("Density")
plt.show()


# %% [markdown]
# ### card_id

# %%
# Top 20 use card_id
n = 20
col = "card_id"
top_n = df[col].value_counts().sort_values(ascending=False).head(n)

fig, ax = plt.subplots(layout="constrained")
sns.barplot(top_n)

ax.tick_params("x", rotation=45)
ax.set_ylabel("Count")
ax.set_title(f"Top {n} most frequent card IDs")
plt.show()


# %%
# Number of card id by fraud count
(
    df.astype({"is_fraud": "int"})
    .groupby("card_id")["is_fraud"]
    .sum()
    .value_counts()
    .sort_index()
    .rename_axis("number_of_frauds")
    .reset_index(name="number_of_cards")
)


# %% [markdown]
# ### merchant_id

# %%
# Top 20 merchant_id
n = 20
col = "merchant_id"
top_n = df[col].value_counts().sort_values(ascending=False).head(n)

fig, ax = plt.subplots(layout="constrained")
sns.barplot(top_n)
for container in ax.containers:
    ax.bar_label(container)
ax.tick_params("x", rotation=45)
ax.set_ylabel("Count")
ax.set_title(f"Top {n} merchants by number of transactions")
plt.show()


# %%
# Number of merchant id by fraud count
(
    df.astype({"is_fraud": "int"})
    .groupby("merchant_id")["is_fraud"]
    .sum()
    .value_counts()
    .sort_index()
    .rename_axis("number_of_frauds")
    .reset_index(name="number_of_merchants")
)


# %% [markdown]
# ### merchant_category_code

# %%
fig, ax = plt.subplots(layout="constrained")

sns.countplot(
    data=df,
    x="merchant_category_code",
    stat="percent",
    hue="is_fraud",
    order=df["merchant_category_code"].value_counts().index,
    ax=ax,
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%")

ax.set_ylabel("Percentage")
ax.set_title("Merchant category by number of transactions")
plt.show()


# %%
# Number of merchant category by fraud count
(
    df.astype({"is_fraud": "int"})
    .groupby("merchant_category_code")["is_fraud"]
    .sum()
    .value_counts()
    .sort_index()
    .rename_axis("number_of_frauds")
    .reset_index(name="number_of_merchant_category_code")
)


# %% [markdown]
# ### merchant_country

# %%
fig, ax = plt.subplots(figsize=(15, 9), layout="constrained")
sns.countplot(
    data=df,
    x="merchant_country",
    stat="percent",
    hue="is_fraud",
    order=df["merchant_country"].value_counts().index,
    ax=ax,
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f%%")

ax.set_ylabel("Percentage")
ax.set_title("Merchant country by number of transactions")
plt.show()


# %%
# Number of merchant country by fraud count
(
    df.astype({"is_fraud": "int"})
    .groupby("merchant_country")["is_fraud"]
    .sum()
    .value_counts()
    .sort_index()
    .rename_axis("number_of_frauds")
    .reset_index(name="number_of_merchant_country")
)


# %% [markdown]
# ### currency

# %%
fig, ax = plt.subplots(figsize=(15, 9), layout="constrained")
sns.countplot(
    data=df,
    x="currency",
    stat="percent",
    hue="is_fraud",
    order=df["currency"].value_counts().index,
    ax=ax,
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%")

ax.set(
    ylabel="Percentage",
    title="Currency by number of transactions",
)
plt.show()


# %%
# Number of currency by fraud count
(
    df.astype({"is_fraud": "int"})
    .groupby("currency")["is_fraud"]
    .sum()
    .value_counts()
    .sort_index()
    .rename_axis("number_of_frauds")
    .reset_index(name="number_of_currency")
)


# %% [markdown]
# ### device_type

# %%
fig, ax = plt.subplots(figsize=(15, 9), layout="constrained")
sns.countplot(
    data=df,
    x="device_type",
    stat="percent",
    hue="is_fraud",
    order=df["device_type"].value_counts().index,
    ax=ax,
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%")

ax.set(
    ylabel="Percentage",
    title="Type of device by number of transactions",
)
plt.show()


# %%
# Number of device type by fraud count
(
    df.astype({"is_fraud": "int"})
    .groupby("device_type")["is_fraud"]
    .sum()
    .value_counts()
    .sort_index()
    .rename_axis("number_of_frauds")
    .reset_index(name="number_of_device_type")
)


# %% [markdown]
# ### entry_mode

# %%
fig, ax = plt.subplots(figsize=(15, 9), layout="constrained")
sns.countplot(
    data=df,
    x="entry_mode",
    stat="percent",
    hue="is_fraud",
    order=df["entry_mode"].value_counts().index,
    ax=ax,
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%")

ax.set(
    ylabel="Percentage",
    title="Entry mode by number of transactions",
)
plt.show()


# %%
# Number of entry mode by fraud count
(
    df.astype({"is_fraud": "int"})
    .groupby("entry_mode")["is_fraud"]
    .sum()
    .value_counts()
    .sort_index()
    .rename_axis("number_of_frauds")
    .reset_index(name="number_of_entry_mode")
)


# %% [markdown]
# ### channel

# %%
fig, ax = plt.subplots(figsize=(15, 9), layout="constrained")
sns.countplot(
    data=df,
    x="channel",
    stat="percent",
    hue="is_fraud",
    order=df["channel"].value_counts().index,
    ax=ax,
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%")

ax.set(
    ylabel="Percentage",
    title="Channel by number of transactions",
)
plt.show()


# %%
# Number of channel by fraud count
(
    df.astype({"is_fraud": "int"})
    .groupby("channel")["is_fraud"]
    .sum()
    .value_counts()
    .sort_index()
    .rename_axis("number_of_frauds")
    .reset_index(name="number_of_channel")
)


# %% [markdown]
# ### is_international

# %%
fig, ax = plt.subplots(figsize=(15, 9), layout="constrained")
sns.countplot(
    data=df,
    x="is_international",
    stat="percent",
    hue="is_fraud",
    order=df["is_international"].value_counts().index,
    ax=ax,
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%")

ax.set(
    ylabel="Percentage",
    title="Number of international transactions",
)
plt.show()


# %%
# Number of international transactions by fraud count
(
    df.astype({"is_fraud": "int"})
    .groupby("is_international")["is_fraud"]
    .sum()
    .value_counts()
    .sort_index()
    .rename_axis("number_of_frauds")
    .reset_index(name="number_of_is_international")
)
