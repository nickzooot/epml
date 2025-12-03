import pathlib

import pandas as pd
from sklearn import datasets


def main() -> None:
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(pathlib.Path("data/raw/iris.csv"), index=False)


if __name__ == "__main__":
    main()
