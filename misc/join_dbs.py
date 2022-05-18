import os
import argparse
from matplotlib import markers
import pandas as pd


def join_dfs(dbs):
    df = pd.DataFrame()
    dfs = []
    for db in dbs:
        # df = df.append(pd.read_csv(os.path.join(args.db_path, db), index_col=0), ignore_index=True)
        dfs.append(pd.read_csv(db, index_col=0))
    df = pd.concat(dfs, ignore_index=True)

    return df

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_path", type=str, default="")
    parser.add_argument("--db_save", type=str, default="dbs_iters.csv")
    
    args = parser.parse_args()
    dbs = os.listdir(args.db_path)

    dbs = [os.path.join(args.db_path, db) for db in dbs]

    # df = pd.DataFrame()
    # dfs = []
    # for db in dbs:
    #     # df = df.append(pd.read_csv(os.path.join(args.db_path, db), index_col=0), ignore_index=True)
    #     dfs.append(pd.read_csv(os.path.join(args.db_path, db), index_col=0))
    # df = pd.concat(dfs, ignore_index=True)

    df = join_dfs(dbs)
    df.to_csv(args.db_save)
    


if __name__ == "__main__":
    main()


