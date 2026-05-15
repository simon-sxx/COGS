from datautil.cls_datautils.data_preprocess_dsads import prep_dsads_data

def prep_cls_data(args):
    print("==================== start prepare data =========================")
    if args.dataset == "dsads":
        train_loader_list, valid_loader, target_loader = prep_dsads_data(args)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented.")
    print("==================== data preparation done ======================")
        
    return train_loader_list, valid_loader, target_loader