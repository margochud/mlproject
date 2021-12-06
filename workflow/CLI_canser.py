import click
from baseline import train_test_split, read_cancer_dataset, plot_precision_recall


@click.command()
@click.argument("path_to_data", type=click.Path())
@click.argument("path_to_plot", type=click.Path())
def plot_cancer(path_to_csv, path_to_data):
    X, y = read_cancer_dataset(path_to_csv)
    X_train, y_train, X_test, y_test = train_test_split(X, y, ratio=0.8)
    plot_precision_recall(X_train, y_train, X_test, y_test, path=path_to_data)


if __name__ == '__main__':
    plot_cancer()
