import pandas as pd


def main():
    df = pd.read_pickle("../resources/CLERC_sample.pkl").head(10)
    print(df)
    pos_para_list = [list(p) for p in df['positive_passages'].tolist()]
    neg_para_list = [list(p) for p in df['negative_passages'].tolist()]
    doc_list = []
    for p in pos_para_list:
        doc_list.extend(p)
    for p in neg_para_list:
        doc_list.extend(p)
    print(doc_list)
    doc_df = pd.DataFrame(doc_list)
    print(doc_df)
    query_df = df
    doc_df.to_pickle("../resources/sample_doc_df.pkl")
    query_df.to_pickle("../resources/sample_query_df.pkl")





if __name__ == '__main__':
    main()