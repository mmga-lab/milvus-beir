import ir_datasets

all_datasets = list(ir_datasets.registry)
for dataset_id in all_datasets:
    dataset = ir_datasets.load(dataset_id)
    print(dataset_id, dataset)
for dataset_id in all_datasets:
    if "beir/scifact/train" in dataset_id:
        dataset = ir_datasets.load(dataset_id)
        if dataset.has_docs() and dataset.has_queries() and dataset.has_qrels():
            for doc in dataset.docs_iter():
                print("doc")
                print(doc)
                break
            for query in dataset.queries_iter():
                print("query")
                print(query)
                break
            for qrel in dataset.qrels_iter():
                print("qrel")
                print(qrel)
                break


