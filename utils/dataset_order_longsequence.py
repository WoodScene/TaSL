def get_dataset_order(dataset_id):
    task_list = [
        "yelp",
        "amazon",
        "dbpedia",
        "yahoo",
        "agnews",
        "MNLI",
        "QQP",
        "RTE",
        "SST-2",
        "WiC",
        "CB",
        "COPA",
        "BoolQA",
        "MultiRC",
        "IMDB"
    ]
    if dataset_id == 1:
        dataset_order = [
            "mnli",
            "cb",
            "wic",
            "copa",
            "qqp",
            "boolqa",
            "rte",
            "imdb",
            "yelp",
            "amazon",
            "sst-2",
            "dbpedia",
            "agnews",
            "multirc",
            "yahoo"
            ]
        
    elif dataset_id == 2:
        dataset_order = [
            "multirc",
            "boolqa",
            "wic",
            "mnli",
            "cb",
            "copa",
            "qqp",
            "rte",
            "imdb",
            "sst-2",
            "dbpedia",
            "agnews",
            "yelp",
            "amazon",
            "yahoo"
        ]
    elif dataset_id == 3:
        dataset_order = [
            "yelp",
            "amazon",
            "mnli",
            "cb",
            "copa",
            "qqp",
            "rte",
            "imdb",
            "sst-2",
            "dbpedia",
            "agnews",
            "yahoo",
            "multirc",
            "boolqa",
            "wic"
        ]
    elif dataset_id == 4:
        dataset_order = [
            "dbpedia",
            "amazon",
            "yahoo",
            "agnews",
        ]
    elif dataset_id == 5:
        dataset_order = [
            "dbpedia",
            "amazon",
            "agnews",
            "yahoo",
        ]
    elif dataset_id == 6:
        dataset_order = [
            "yahoo",
            "amazon",
            "agnews",
            "dbpedia",
        ]
    else:
        raise

    return dataset_order