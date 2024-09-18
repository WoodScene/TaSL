def get_dataset_order(dataset_id):

    if dataset_id == 1:
        dataset_order = [
            "task1572",
            "task363",
            "task1290",
            "task181",
            "task002",
            "task1510",
            "task639",
            "task1729",
            "task073",
            "task1590",
            "task748",
            "task511",
            "task591",
            "task1687",
            "task875"
            ]
        
    elif dataset_id == 2:
        dataset_order = [
            "task748",
            "task073",
            "task1590",
            "task639",
            "task1572",
            "task1687",
            "task591",
            "task363",
            "task1510",
            "task1729",
            "task181",
            "task511",
            "task002",
            "task1290",
            "task875"
        ]
    
    else:
        raise

    return dataset_order