def get_data_info(dataset_name):
    dataset_para = dict()
    # dataset_para['ascad'] = {  #
    #                              'dataset_name': 'ascad',
    #                              'feature_num': 700,
    #                              'leakage_model': 'ID',
    #                              'classification': 256,
    #                              'true_key': 0xe0,
    #                              'trs_fname': 'ascad_trs.npy',
    #                              'label_fname': 'ascad_label.npy',
    #                              'plain_fname': 'ascad_plain.npy',
    #                              'ascad_cnn_block_anti_bn_dense_input': 10752,
    #                              'ascad_cnn_dense_input': 10752}
    # dataset_para['ascad_20k'] = {
    #                             'dataset_name': 'ascad_20k',
    #                             'feature_num': 700,
    #                             'leakage_model': 'ID',
    #                             'classification': 256,
    #                             'true_key': 0xe0,
    #                             'trs_fname': 'ascad_20k_trs.npy',
    #                             'label_fname': 'ascad_20k_label.npy',
    #                             'plain_fname': 'ascad_20k_plain.npy',
    #                             'ascad_cnn_block_anti_bn_dense_input': 10752,
    #                             'ascad_cnn_dense_input': 10752}
    # dataset_para['ascad_30k'] = {  # ?
    #                              'dataset_name': 'ascad_30k',
    #                              'feature_num': 700,
    #                              'leakage_model': 'ID',
    #                              'classification': 256,
    #                              'true_key': 0xe0,
    #                              'trs_fname': 'ascad_30k_trs.npy',
    #                              'label_fname': 'ascad_30k_label.npy',
    #                              'plain_fname': 'ascad_30k_plain.npy',
    #                              'ascad_cnn_block_anti_bn_dense_input': 10752,
    #                              'ascad_cnn_dense_input': 10752}
    dataset_para['EM'] = {  # filter = 1, shift = 5, cut = 5, 目前最好的是都用
                          'dataset_name': 'EM',
                          'feature_num': 90,
                          'leakage_model': 'HD',
                          'classification': 9,
                          'true_key': 19,
                          'trs_fname': 'EM_trs.npy',
                          'label_fname': 'EM_label.npy',
                          'plain_fname': 'EM_cipher.npy',
                          'ascad_cnn_block_anti_bn_dense_input': 1024,
                          'ascad_cnn_dense_input': 1024}
    # dataset_para['EM_500'] = {  # filter = 1, shift = 5, cut = 5, 目前最好的是都用
    #                         'dataset_name': 'EM_500',
    #                         'feature_num': 500,
    #                         'leakage_model': 'HD',
    #                         'classification': 9,
    #                         'true_key': 19,
    #                         'trs_fname': 'EM_trs_500.npy',
    #                         'label_fname': 'EM_label.npy',
    #                         'plain_fname': 'EM_cipher.npy',
    #                         'ascad_cnn_block_anti_bn_dense_input': 7680,
    #                         'ascad_cnn_dense_input': 7680}
    return dataset_para[dataset_name]
